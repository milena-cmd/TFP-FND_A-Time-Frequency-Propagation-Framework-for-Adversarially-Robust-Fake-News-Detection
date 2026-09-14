"""Frequency-aware content representation, Section "Frequency-Aware Content Representation".

  s_j  = masked mean pooling of BERT's contextual token representations   (Eq. semantic_pooling)
  F_j  = rFFT along the token-position axis, applied per hidden dimension (Eq. after "Motivation")
  E_j  = g_j * s_j^p + (1 - g_j) * u_j^p                                  (Eq. content_fusion)

`shuffle_positions=True` implements the ablation described in the same section: token
positions are permuted before the DFT to test whether the frequency branch captures
real sequential structure (expect a performance drop vs. the unshuffled variant).
"""
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class FrequencyAwareContentView(nn.Module):
    def __init__(self, bert_name: str = "bert-base-uncased", bert_hidden: int = 768,
                 content_hidden: int = 256, freq_attn_heads: int = 4, dropout: float = 0.1,
                 use_frequency_branch: bool = True, use_semantic_branch: bool = True):
        super().__init__()
        if not (use_frequency_branch or use_semantic_branch):
            raise ValueError("At least one of the semantic/frequency branches must be enabled.")
        self.use_frequency_branch = use_frequency_branch
        self.use_semantic_branch = use_semantic_branch

        self.bert = AutoModel.from_pretrained(bert_name)
        self.bert_hidden = bert_hidden

        if use_semantic_branch:
            self.semantic_proj = nn.Sequential(
                nn.Linear(bert_hidden, content_hidden), nn.GELU(), nn.Dropout(dropout),
            )
        if use_frequency_branch:
            self.freq_input_proj = nn.Linear(bert_hidden, content_hidden)
            self.freq_attn = nn.MultiheadAttention(
                content_hidden, freq_attn_heads, dropout=dropout, batch_first=True,
            )
            self.freq_norm = nn.LayerNorm(content_hidden)
            self.freq_proj = nn.Sequential(
                nn.Linear(content_hidden, content_hidden), nn.GELU(), nn.Dropout(dropout),
            )
        if use_semantic_branch and use_frequency_branch:
            self.gate = nn.Sequential(nn.Linear(2 * content_hidden, content_hidden), nn.Sigmoid())

        self.content_hidden = content_hidden

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                shuffle_positions: bool = False) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, bert_hidden]
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        s_p = None
        if self.use_semantic_branch:
            s = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)  # Eq. semantic_pooling
            s_p = self.semantic_proj(s)

        u_p = None
        if self.use_frequency_branch:
            h = hidden * mask  # zero out padding before the transform
            if shuffle_positions:
                perm = torch.randperm(h.size(1), device=h.device)
                h = h[:, perm, :]
            # real-valued FFT along the token-position axis, independently per hidden dim
            freq = torch.fft.rfft(h, dim=1)               # [B, L//2+1, bert_hidden] complex
            freq_mag = torch.abs(freq)                     # magnitude spectrum
            freq_mag = self.freq_input_proj(freq_mag)       # [B, freq_bins, content_hidden]
            attn_out, _ = self.freq_attn(freq_mag, freq_mag, freq_mag)
            freq_repr = self.freq_norm(freq_mag + attn_out).mean(dim=1)  # pool over frequency bins
            u_p = self.freq_proj(freq_repr)

        if s_p is not None and u_p is not None:
            g = self.gate(torch.cat([s_p, u_p], dim=-1))
            return g * s_p + (1 - g) * u_p  # Eq. content_fusion
        return s_p if s_p is not None else u_p
