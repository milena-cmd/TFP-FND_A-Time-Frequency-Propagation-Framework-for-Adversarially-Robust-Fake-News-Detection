"""Temporal diffusion view, Section "Temporal Diffusion Representation".

Builds a per-event feature (user embedding, normalized elapsed time delta_t and
propagation depth), processes the sequence with causally-masked self-attention, and
combines the attention-pooled sequence with four aggregate descriptors (propagation
rate, early-activity concentration, max depth, temporal dispersion) into H_temp.

Causality: the causal mask constrains what one event's attention can see (nothing
after its own timestamp), and all features are computed from the interactions
observed before the prediction deadline only. The optional "subsequent repost count"
feature requires information from later events and is disabled
(`use_future_repost_count: false`), which is the setting used in all experiments.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from src.data.historical import HistoricalContext, UNK_USER


def _event_ts(e: dict, default: float = 0.0) -> float:
    ts = e.get("ts")
    try:
        return float(ts) if ts is not None else default
    except (TypeError, ValueError):
        return default


class TemporalDiffusionView(nn.Module):
    def __init__(self, user_vocab_size: int, user_embed_dim: int = 64,
                 hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, use_future_repost_count: bool = False):
        super().__init__()
        self.use_future_repost_count = use_future_repost_count
        self.user_embed = nn.Embedding(user_vocab_size, user_embed_dim, padding_idx=0)

        n_scalar_feats = 3 if use_future_repost_count else 2  # delta_t, depth, [future_repost]
        self.event_proj = nn.Linear(user_embed_dim + n_scalar_feats, hidden_dim)
        self.pos_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn_pool = nn.MultiheadAttention(hidden_dim, num_heads=1, dropout=dropout, batch_first=True)

        self.aggregate_proj = nn.Linear(4, hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.hidden_dim = hidden_dim

    def _featurize(self, cascade: List[dict], historical_ctx: HistoricalContext,
                   tau_seconds: float, device: torch.device):
        n = len(cascade)
        if n == 0:
            user_idx = torch.tensor([historical_ctx.user_index(UNK_USER)], dtype=torch.long, device=device)
            scalars = torch.zeros(1, 3 if self.use_future_repost_count else 2, device=device)
            return user_idx, scalars, 0

        events_sorted = sorted(cascade, key=lambda e: _event_ts(e))
        user_idx = torch.tensor(
            [historical_ctx.user_index(e.get("user", UNK_USER)) for e in events_sorted],
            dtype=torch.long, device=device,
        )
        max_depth = max((e.get("depth") or 0) for e in events_sorted) or 1
        deltas = [min(1.0, max(0.0, _event_ts(e) / tau_seconds)) if tau_seconds > 0 else 0.0
                  for e in events_sorted]
        depths = [(e.get("depth") or 0) / max_depth for e in events_sorted]

        if self.use_future_repost_count:
            ts_list = [_event_ts(e) for e in events_sorted]
            future_counts = []
            for i, t in enumerate(ts_list):
                future_counts.append(sum(1 for t2 in ts_list if t2 > t))
            max_future = max(future_counts) or 1
            future_counts = [c / max_future for c in future_counts]
            scalars = torch.tensor(list(zip(deltas, depths, future_counts)), dtype=torch.float, device=device)
        else:
            scalars = torch.tensor(list(zip(deltas, depths)), dtype=torch.float, device=device)

        return user_idx, scalars, n

    def _aggregate_descriptors(self, cascade: List[dict], tau_seconds: float,
                                early_window_seconds: float = 300.0) -> List[float]:
        n = len(cascade)
        if n == 0:
            return [0.0, 0.0, 0.0, 0.0]
        ts_list = sorted(_event_ts(e) for e in cascade)
        rate = n / max(tau_seconds, 1.0)
        early = sum(1 for t in ts_list if t <= early_window_seconds) / n
        max_depth = max((e.get("depth") or 0) for e in cascade)
        dispersion = float(torch.var(torch.tensor(ts_list, dtype=torch.float))) if n > 1 else 0.0
        return [rate, early, float(max_depth), dispersion]

    def forward(self, cascades: List[List[dict]], historical_ctx: HistoricalContext,
                tau_seconds: float, device: torch.device) -> torch.Tensor:
        batch_size = len(cascades)
        featurized = [self._featurize(c, historical_ctx, tau_seconds, device) for c in cascades]
        max_len = max(1, max(f[2] for f in featurized))

        user_idx_batch = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        scalar_dim = 3 if self.use_future_repost_count else 2
        scalars_batch = torch.zeros(batch_size, max_len, scalar_dim, device=device)
        key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)  # True = pad
        causal_mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool, device=device), diagonal=1)

        for i, (user_idx, scalars, n) in enumerate(featurized):
            length = max(n, 1)
            user_idx_batch[i, :length] = user_idx[:length]
            scalars_batch[i, :length] = scalars[:length]
            key_padding_mask[i, :length] = False

        user_embeds = self.user_embed(user_idx_batch)
        event_feats = torch.cat([user_embeds, scalars_batch], dim=-1)
        event_feats = self.pos_norm(self.event_proj(event_feats))

        encoded = self.encoder(
            event_feats, mask=causal_mask, src_key_padding_mask=key_padding_mask,
        )

        query = self.attn_pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.attn_pool(query, encoded, encoded, key_padding_mask=key_padding_mask)
        pooled = pooled.squeeze(1)

        agg = torch.tensor(
            [self._aggregate_descriptors(c, tau_seconds) for c in cascades],
            dtype=torch.float, device=device,
        )
        agg_repr = self.aggregate_proj(agg)

        return self.output_proj(torch.cat([pooled, agg_repr], dim=-1))
