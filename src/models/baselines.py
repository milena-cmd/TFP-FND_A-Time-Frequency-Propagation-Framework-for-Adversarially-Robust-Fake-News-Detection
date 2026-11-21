"""Baseline models, Section "Baseline Methods" / Table "baseline_sources".

`build_model(name, config, user_vocab_size)` is the single entry point used by
src/training/train.py; every baseline is either:
  (a) TFPFND with some views/branches disabled via TFPFNDConfig (BERT-base, BERT+DFT,
      BERT+Temporal, BERT+Bi-GCN/HG-SL "Ours" concatenation rows, and every ablation in
      Table "ablation_arch"), or
  (b) a standalone reimplementation for methods whose published architecture doesn't
      map cleanly onto the TFP-FND view decomposition (Bi-GCN, HG-SL, LeHT).

Bi-GCN, HG-SL and LeHT below are best-effort reimplementations from their papers'
architectural descriptions (Bian et al. 2020; Sun et al. 2023; Wei et al. 2026) --
verify against the authors' released code before citing an exact head-to-head number.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from src.data.historical import HistoricalContext, UNK_USER
from src.models.content_view import FrequencyAwareContentView
from src.models.propagation_view import PropagationView, RELATIONS
from src.models.temporal_view import TemporalDiffusionView
from src.models.tfp_fnd import TFPFND, TFPFNDConfig, _ConcatFusion, _FlexibleFusion


def build_model(name: str, base_config: TFPFNDConfig, user_vocab_size: int, num_classes: int = 2) -> nn.Module:
    name = name.lower()

    if name == "tfp_fnd":
        return TFPFND(base_config, user_vocab_size, num_classes)

    if name == "tfp_fnd_clean_only":
        # same architecture, trained on clean instances only (see src/training/train.py).
        return TFPFND(base_config, user_vocab_size, num_classes)

    if name == "tfp_fnd_wo_robust":
        # identical architecture; lambda_robust=lambda_view=0 is set in the training
        # config/CLI, not here -- this name exists so scripts can select it explicitly.
        return TFPFND(base_config, user_vocab_size, num_classes)

    if name == "bert_base":
        cfg = _clone(base_config, use_frequency_branch=False, use_temporal_view=False,
                     use_propagation_view=False)
        return TFPFND(cfg, user_vocab_size, num_classes)

    if name == "bert_dft":
        cfg = _clone(base_config, use_temporal_view=False, use_propagation_view=False)
        return TFPFND(cfg, user_vocab_size, num_classes)

    if name == "erfd":
        # Fourier-domain (frequency-only) content encoder, no diffusion information.
        cfg = _clone(base_config, use_semantic_branch=False, use_temporal_view=False,
                     use_propagation_view=False)
        return TFPFND(cfg, user_vocab_size, num_classes)

    # Concatenation-fusion baselines.
    if name == "bert_temporal":
        cfg = _clone(base_config, use_frequency_branch=False, use_propagation_view=False,
                     use_adaptive_fusion=False)
        return TFPFND(cfg, user_vocab_size, num_classes)

    if name == "bert_bigcn":
        cfg = _clone(base_config, use_frequency_branch=False, use_temporal_view=False,
                     use_adaptive_fusion=False,
                     enabled_relations=["reply_thread"])  # closest single-relation analogue to a cascade graph
        return TFPFND(cfg, user_vocab_size, num_classes)

    if name == "bert_hgsl":
        cfg = _clone(base_config, use_frequency_branch=False, use_temporal_view=False,
                     use_adaptive_fusion=False)
        return TFPFND(cfg, user_vocab_size, num_classes)

    if name == "bert_dft_temporal":
        cfg = _clone(base_config, use_propagation_view=False, use_adaptive_fusion=False)
        return TFPFND(cfg, user_vocab_size, num_classes)

    if name == "bert_dft_hypergraph":
        cfg = _clone(base_config, use_temporal_view=False, use_adaptive_fusion=False)
        return TFPFND(cfg, user_vocab_size, num_classes)

    # --- architectural ablations, Table "ablation_arch" ---
    if name == "tfp_fnd_wo_frequency":
        return TFPFND(_clone(base_config, use_frequency_branch=False), user_vocab_size, num_classes)
    if name == "tfp_fnd_wo_semantic":
        return TFPFND(_clone(base_config, use_semantic_branch=False), user_vocab_size, num_classes)
    if name == "tfp_fnd_wo_temporal":
        return TFPFND(_clone(base_config, use_temporal_view=False), user_vocab_size, num_classes)
    if name == "tfp_fnd_wo_hypergraph":
        return TFPFND(_clone(base_config, use_propagation_view=False), user_vocab_size, num_classes)
    if name == "tfp_fnd_wo_adaptive_fusion":
        return TFPFND(_clone(base_config, use_adaptive_fusion=False), user_vocab_size, num_classes)
    if name == "tfp_fnd_pairwise_propagation":
        # Section "Higher-Order Propagation Ablation": pairwise GCN instead of hypergraph.
        return TFPFNDWithPairwiseGCN(base_config, user_vocab_size, num_classes)

    if name == "bigcn":
        return BiGCN(base_config, user_vocab_size, num_classes)
    if name == "hgsl":
        return HGSLApprox(base_config, user_vocab_size, num_classes)
    if name == "leht":
        return LeHTApprox(base_config, user_vocab_size, num_classes)

    raise ValueError(f"Unknown model name: {name}")


def _clone(config: TFPFNDConfig, **overrides) -> TFPFNDConfig:
    fields = {k: getattr(config, k) for k in config.__dataclass_fields__}
    fields.update(overrides)
    return TFPFNDConfig(**fields)


# ---------------------------------------------------------------------------
# Bi-GCN (Bian et al., 2020) -- bidirectional GCN over the reply tree.
# Pure propagation baseline: no text, no temporal features beyond tree structure.
# ---------------------------------------------------------------------------
class BiGCN(nn.Module):
    def __init__(self, config: TFPFNDConfig, user_vocab_size: int, num_classes: int = 2):
        super().__init__()
        self.user_embed = nn.Embedding(user_vocab_size, config.user_embed_dim, padding_idx=0)
        gcn_hidden = config.hypergraph_hidden
        self.td_gcn1 = nn.Linear(config.user_embed_dim, gcn_hidden)
        self.td_gcn2 = nn.Linear(gcn_hidden, gcn_hidden)
        self.bu_gcn1 = nn.Linear(config.user_embed_dim, gcn_hidden)
        self.bu_gcn2 = nn.Linear(gcn_hidden, gcn_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(2 * gcn_hidden, config.fusion_hidden), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.fusion_hidden, num_classes),
        )

    def _adjacency(self, cascade: List[dict], source_id: str, users: List[str]) -> torch.Tensor:
        idx = {u: i for i, u in enumerate(users)}
        n = len(users)
        adj = torch.eye(n)
        id_to_user = {source_id: source_id}
        for e in cascade:
            id_to_user[e["id"]] = e.get("user", "unknown")
        for e in cascade:
            parent_user = id_to_user.get(e["parent"], source_id)
            child_user = e.get("user", "unknown")
            if parent_user in idx and child_user in idx:
                adj[idx[parent_user], idx[child_user]] = 1
        return adj

    def _instance_forward(self, cascade: List[dict], source_id: str, historical_ctx: HistoricalContext,
                           device: torch.device) -> torch.Tensor:
        users = sorted({source_id} | {e.get("user", "unknown") for e in cascade}) or [UNK_USER]
        idx = torch.tensor([historical_ctx.user_index(u) for u in users], device=device)
        x = self.user_embed(idx)
        adj = self._adjacency(cascade, source_id, users).to(device)
        deg = adj.sum(-1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        td = torch.relu(self.td_gcn1(adj_norm @ x))
        td = torch.relu(self.td_gcn2(adj_norm @ td)).mean(dim=0)

        adj_t_norm = adj.t() / adj.t().sum(-1, keepdim=True).clamp(min=1)
        bu = torch.relu(self.bu_gcn1(adj_t_norm @ x))
        bu = torch.relu(self.bu_gcn2(adj_t_norm @ bu)).mean(dim=0)

        return torch.cat([td, bu], dim=-1)

    def forward(self, batch_condition: Dict, historical_ctx: HistoricalContext,
                tau_seconds: float, device: torch.device) -> Dict[str, torch.Tensor]:
        cascades = batch_condition["cascades"]
        source_ids = [f"__src_{i}__" for i in range(len(cascades))]  # source acts as its own pseudo-user
        reprs = torch.stack([
            self._instance_forward(c, sid, historical_ctx, device) for c, sid in zip(cascades, source_ids)
        ])
        logits = self.classifier(reprs)
        beta = torch.ones(logits.size(0), 1, device=device)
        return {"logits": logits, "fused": reprs, "beta": beta, "v_prime": reprs.unsqueeze(1)}


# ---------------------------------------------------------------------------
# HG-SL (Sun et al., 2023) approximation -- global (historical) + local (temporal/
# structural) user spreading behavior via the shared hypergraph propagation view.
# ---------------------------------------------------------------------------
class HGSLApprox(nn.Module):
    def __init__(self, config: TFPFNDConfig, user_vocab_size: int, num_classes: int = 2):
        super().__init__()
        self.propagation_view = PropagationView(
            user_vocab_size=user_vocab_size, user_embed_dim=config.user_embed_dim,
            hidden_dim=config.hypergraph_hidden, output_dim=config.propagation_hidden,
            dropout=config.dropout, enabled_relations=RELATIONS,
        )
        self.global_activity_proj = nn.Linear(1, config.propagation_hidden)
        self.classifier = nn.Sequential(
            nn.Linear(2 * config.propagation_hidden, config.fusion_hidden), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.fusion_hidden, num_classes),
        )

    def forward(self, batch_condition: Dict, historical_ctx: HistoricalContext,
                tau_seconds: float, device: torch.device) -> Dict[str, torch.Tensor]:
        local_repr = self.propagation_view(batch_condition["relations"], historical_ctx, device)

        global_activity = []
        for relations in batch_condition["relations"]:
            users = set()
            for edges in relations.values():
                for e in edges:
                    users.update(e["users"])
            avg_activity = sum(historical_ctx.user_activity.get(u, 0) for u in users) / max(1, len(users))
            global_activity.append([avg_activity])
        global_activity = torch.tensor(global_activity, dtype=torch.float, device=device)
        global_repr = self.global_activity_proj(global_activity)

        fused = torch.cat([local_repr, global_repr], dim=-1)
        logits = self.classifier(fused)
        beta = torch.full((logits.size(0), 2), 0.5, device=device)
        v_prime = torch.stack([local_repr, global_repr], dim=1)
        return {"logits": logits, "fused": fused, "beta": beta, "v_prime": v_prime}


# ---------------------------------------------------------------------------
# LeHT (Wei et al., 2026) approximation -- content semantics + collective hypergraph
# propagation, fused by concatenation (no frequency branch, no dedicated temporal view,
# matching Table "related_comparison": Content + Propagation + Hypergraph only).
# ---------------------------------------------------------------------------
class LeHTApprox(nn.Module):
    def __init__(self, config: TFPFNDConfig, user_vocab_size: int, num_classes: int = 2):
        super().__init__()
        self.content_view = FrequencyAwareContentView(
            bert_name=config.bert_name, bert_hidden=config.bert_hidden,
            content_hidden=config.content_hidden, dropout=config.dropout,
            use_frequency_branch=False, use_semantic_branch=True,
        )
        self.propagation_view = PropagationView(
            user_vocab_size=user_vocab_size, user_embed_dim=config.user_embed_dim,
            hidden_dim=config.hypergraph_hidden, output_dim=config.propagation_hidden,
            dropout=config.dropout, enabled_relations=RELATIONS,
        )
        self.fusion = _ConcatFusion([config.content_hidden, config.propagation_hidden],
                                     config.fusion_hidden, dropout=config.dropout)
        self.classifier = nn.Linear(config.fusion_hidden, num_classes)

    def forward(self, batch_condition: Dict, historical_ctx: HistoricalContext,
                tau_seconds: float, device: torch.device) -> Dict[str, torch.Tensor]:
        content_repr = self.content_view(
            batch_condition["input_ids"].to(device), batch_condition["attention_mask"].to(device),
        )
        propagation_repr = self.propagation_view(batch_condition["relations"], historical_ctx, device)
        fused, beta, v_prime = self.fusion([content_repr, propagation_repr])
        logits = self.classifier(fused)
        return {"logits": logits, "fused": fused, "beta": beta, "v_prime": v_prime}


# ---------------------------------------------------------------------------
# Higher-order propagation ablation (Section "Higher-Order Propagation Ablation"):
# swap the hypergraph module for a pairwise GCN, keep everything else identical.
# ---------------------------------------------------------------------------
class TFPFNDWithPairwiseGCN(nn.Module):
    def __init__(self, config: TFPFNDConfig, user_vocab_size: int, num_classes: int = 2):
        super().__init__()
        self.content_view = FrequencyAwareContentView(
            bert_name=config.bert_name, bert_hidden=config.bert_hidden,
            content_hidden=config.content_hidden, dropout=config.dropout,
            use_frequency_branch=config.use_frequency_branch, use_semantic_branch=config.use_semantic_branch,
        )
        self.temporal_view = TemporalDiffusionView(
            user_vocab_size=user_vocab_size, user_embed_dim=config.user_embed_dim,
            hidden_dim=config.temporal_hidden, num_layers=config.temporal_layers,
            num_heads=config.temporal_heads, dropout=config.dropout,
            use_future_repost_count=config.use_future_repost_count,
        )
        self.pairwise_gcn = BiGCN(config, user_vocab_size, num_classes=config.propagation_hidden)
        self.pairwise_proj = nn.Linear(2 * config.hypergraph_hidden, config.propagation_hidden)
        self.fusion = _FlexibleFusion(
            [config.content_hidden, config.temporal_hidden, config.propagation_hidden],
            config.fusion_hidden, dropout=config.dropout,
        )
        self.classifier = nn.Linear(config.fusion_hidden, num_classes)

    def forward(self, batch_condition: Dict, historical_ctx: HistoricalContext,
                tau_seconds: float, device: torch.device) -> Dict[str, torch.Tensor]:
        content_repr = self.content_view(
            batch_condition["input_ids"].to(device), batch_condition["attention_mask"].to(device),
        )
        temporal_repr = self.temporal_view(batch_condition["cascades"], historical_ctx, tau_seconds, device)

        cascades = batch_condition["cascades"]
        pairwise_reprs = torch.stack([
            self.pairwise_gcn._instance_forward(c, f"__src_{i}__", historical_ctx, device)
            for i, c in enumerate(cascades)
        ])
        propagation_repr = self.pairwise_proj(pairwise_reprs)

        fused, beta, v_prime = self.fusion([content_repr, temporal_repr, propagation_repr])
        logits = self.classifier(fused)
        return {"logits": logits, "fused": fused, "beta": beta, "v_prime": v_prime}
