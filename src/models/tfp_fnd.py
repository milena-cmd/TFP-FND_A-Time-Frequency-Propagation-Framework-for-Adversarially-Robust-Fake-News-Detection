"""TFP-FND: full model, Section "Summary of the TFP-FND Pipeline".

forward() runs one of the four conditions (clean / content / propagation / joint) end
to end and returns logits + the fused representation R_j + per-view contextualized
representations V'_j (needed by L_view) + reliability weights beta (for RQ4). The
training loop calls forward() once per condition per batch and combines the four
outputs with the losses in src/training/losses.py.

`ablation` toggles let src/models/baselines.py and the RQ2/RQ6 scripts reuse this same
class instead of hand-duplicating the architecture for every ablation row in Table
"ablation_arch".
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.data.historical import HistoricalContext
from src.models.content_view import FrequencyAwareContentView
from src.models.fusion import AdaptiveCrossViewFusion
from src.models.propagation_view import PropagationView, RELATIONS
from src.models.temporal_view import TemporalDiffusionView


@dataclass
class TFPFNDConfig:
    bert_name: str = "bert-base-uncased"
    bert_hidden: int = 768
    content_hidden: int = 256
    temporal_layers: int = 2
    temporal_heads: int = 4
    temporal_hidden: int = 256
    hypergraph_hidden: int = 128
    propagation_hidden: int = 256
    fusion_hidden: int = 256
    dropout: float = 0.1
    user_embed_dim: int = 64
    use_future_repost_count: bool = False

    # architectural ablation switches (Table "ablation_arch")
    use_frequency_branch: bool = True
    use_semantic_branch: bool = True
    use_temporal_view: bool = True
    use_propagation_view: bool = True
    use_adaptive_fusion: bool = True  # False -> plain concatenation instead of reliability weighting
    enabled_relations: List[str] = field(default_factory=lambda: list(RELATIONS))


class TFPFND(nn.Module):
    def __init__(self, config: TFPFNDConfig, user_vocab_size: int, num_classes: int = 2):
        super().__init__()
        self.config = config

        if config.use_semantic_branch or config.use_frequency_branch:
            self.content_view = FrequencyAwareContentView(
                bert_name=config.bert_name, bert_hidden=config.bert_hidden,
                content_hidden=config.content_hidden, dropout=config.dropout,
                use_frequency_branch=config.use_frequency_branch,
                use_semantic_branch=config.use_semantic_branch,
            )
        else:
            raise ValueError("At least one content sub-branch must remain enabled.")

        if config.use_temporal_view:
            self.temporal_view = TemporalDiffusionView(
                user_vocab_size=user_vocab_size, user_embed_dim=config.user_embed_dim,
                hidden_dim=config.temporal_hidden, num_layers=config.temporal_layers,
                num_heads=config.temporal_heads, dropout=config.dropout,
                use_future_repost_count=config.use_future_repost_count,
            )
        else:
            self.temporal_view = None

        if config.use_propagation_view:
            self.propagation_view = PropagationView(
                user_vocab_size=user_vocab_size, user_embed_dim=config.user_embed_dim,
                hidden_dim=config.hypergraph_hidden, output_dim=config.propagation_hidden,
                dropout=config.dropout, enabled_relations=config.enabled_relations,
            )
        else:
            self.propagation_view = None

        active_dims = [config.content_hidden]
        if self.temporal_view is not None:
            active_dims.append(config.temporal_hidden)
        if self.propagation_view is not None:
            active_dims.append(config.propagation_hidden)

        if config.use_adaptive_fusion and len(active_dims) > 1:
            # AdaptiveCrossViewFusion is defined for exactly 3 views; when an ablation
            # disables one, fall back to a 2-view or 1-view version of the same module
            # so "w/o temporal branch" etc. still routes through *a* fusion mechanism.
            self.fusion = _FlexibleFusion(active_dims, config.fusion_hidden, dropout=config.dropout)
        else:
            self.fusion = _ConcatFusion(active_dims, config.fusion_hidden, dropout=config.dropout)

        self.classifier = nn.Linear(config.fusion_hidden, num_classes)

    def forward(self, batch_condition: Dict, historical_ctx: HistoricalContext,
                tau_seconds: float, device: torch.device) -> Dict[str, torch.Tensor]:
        content_repr = self.content_view(
            batch_condition["input_ids"].to(device), batch_condition["attention_mask"].to(device),
        )

        views = [content_repr]
        if self.temporal_view is not None:
            temporal_repr = self.temporal_view(
                batch_condition["cascades"], historical_ctx, tau_seconds, device,
            )
            views.append(temporal_repr)
        if self.propagation_view is not None:
            propagation_repr = self.propagation_view(
                batch_condition["relations"], historical_ctx, device,
            )
            views.append(propagation_repr)

        fused, beta, v_prime = self.fusion(views)
        logits = self.classifier(fused)
        return {"logits": logits, "fused": fused, "beta": beta, "v_prime": v_prime}


class _FlexibleFusion(nn.Module):
    """AdaptiveCrossViewFusion generalized to 1-3 views (for architectural ablations)."""

    def __init__(self, view_dims: List[int], fusion_hidden: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_views = len(view_dims)
        self.projs = nn.ModuleList([nn.Linear(d, fusion_hidden) for d in view_dims])
        if self.n_views > 1:
            self.attn = nn.MultiheadAttention(fusion_hidden, num_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(fusion_hidden)
            self.reliability_proj = nn.Linear(fusion_hidden, 1)

    def forward(self, views: List[torch.Tensor]):
        projected = [proj(v) for proj, v in zip(self.projs, views)]
        if self.n_views == 1:
            v = projected[0]
            beta = torch.ones(v.size(0), 1, device=v.device)
            return v, beta, v.unsqueeze(1)

        v = torch.stack(projected, dim=1)
        attn_out, _ = self.attn(v, v, v)
        v_prime = self.norm(v + attn_out)
        scores = self.reliability_proj(v_prime).squeeze(-1)
        beta = torch.softmax(scores, dim=-1)
        fused = (beta.unsqueeze(-1) * v_prime).sum(dim=1)
        return fused, beta, v_prime


class _ConcatFusion(nn.Module):
    """'w/o adaptive fusion' ablation: plain concatenation + linear projection."""

    def __init__(self, view_dims: List[int], fusion_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(sum(view_dims), fusion_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.n_views = len(view_dims)

    def forward(self, views: List[torch.Tensor]):
        fused = self.proj(torch.cat(views, dim=-1))
        beta = torch.full((fused.size(0), self.n_views), 1.0 / self.n_views, device=fused.device)
        v_prime = torch.stack(views, dim=1)  # not contextualized, but kept for shape compatibility
        return fused, beta, v_prime
