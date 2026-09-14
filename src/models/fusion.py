"""Adaptive cross-view fusion, Section "Adaptive Cross-View Fusion".

  V_j[m,:]   = phi_m(W_m X_j^(m) + b_m)              (Eq. view_projection)
  V'_j       = ViewSelfAttention(V_j)                 (contextualization across the 3 views)
  s_{j,m}    = w_r^T V'_j[m,:]                        (Eq. view_score)
  beta_{j,m} = softmax_m(s_{j,m})                     (Eq. reliability_weights)
  R_j        = sum_m beta_{j,m} V'_j[m,:]             (Eq. fused_representation)

Returns both R_j (for the classification head) and beta (for RQ4's reliability
analysis / logging) plus V'_j (needed by the view-level consistency loss).
"""
from typing import Tuple

import torch
import torch.nn as nn


class AdaptiveCrossViewFusion(nn.Module):
    def __init__(self, content_dim: int, temporal_dim: int, propagation_dim: int,
                 fusion_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj_content = nn.Linear(content_dim, fusion_dim)
        self.proj_temporal = nn.Linear(temporal_dim, fusion_dim)
        self.proj_propagation = nn.Linear(propagation_dim, fusion_dim)

        self.view_attn = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.view_norm = nn.LayerNorm(fusion_dim)
        self.reliability_proj = nn.Linear(fusion_dim, 1)
        self.fusion_dim = fusion_dim

    def forward(self, content_repr: torch.Tensor, temporal_repr: torch.Tensor,
                propagation_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v_content = self.proj_content(content_repr)
        v_temporal = self.proj_temporal(temporal_repr)
        v_propagation = self.proj_propagation(propagation_repr)

        v = torch.stack([v_content, v_temporal, v_propagation], dim=1)  # [B, 3, fusion_dim]
        attn_out, _ = self.view_attn(v, v, v)
        v_prime = self.view_norm(v + attn_out)  # [B, 3, fusion_dim] -- V'_j

        scores = self.reliability_proj(v_prime).squeeze(-1)  # [B, 3]  -- s_{j,m}
        beta = torch.softmax(scores, dim=-1)                  # [B, 3]  -- beta_{j,m}

        fused = (beta.unsqueeze(-1) * v_prime).sum(dim=1)      # [B, fusion_dim] -- R_j
        return fused, beta, v_prime
