"""Multi-relational propagation view, Section "Multi-Relational Propagation Representation".

For each relation type (co_propagation, temporal_cluster, reply_thread, community):
  1. Mean-aggregate member-user embeddings (+ coordination descriptors) into a
     hyperedge representation.
  2. Attention-based message passing back to member nodes: softmax over the dot
     product of user/hyperedge query-key projections.
  3. Aggregate the resulting node representations over the cascade -> relation-specific
     propagation representation.
Relation branches are then adaptively combined (learned weights conditioned on the
relation representations themselves) into the final P_j.

This is implemented as a per-instance loop (hyperedges vary in size/count per sample),
which is simple and correct but not vectorized across the batch; for large-scale runs,
consider porting to a PyTorch Geometric HypergraphConv for throughput.
"""
from typing import Dict, List

import torch
import torch.nn as nn

from src.data.historical import HistoricalContext, UNK_USER

RELATIONS = ["co_propagation", "temporal_cluster", "reply_thread", "community"]


class RelationBranch(nn.Module):
    def __init__(self, user_embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hyperedge_proj = nn.Linear(user_embed_dim + 3, hidden_dim)  # +3 coordination feats
        self.query_proj = nn.Linear(user_embed_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_embeds: torch.Tensor, edges: List[dict], user_index_map: Dict[str, int],
                device: torch.device) -> torch.Tensor:
        """user_embeds: [n_unique_users, embed_dim] for this instance's node set.
        Returns a single pooled representation for this relation branch, this instance."""
        if not edges:
            return torch.zeros(self.out_norm.normalized_shape[0], device=device)

        hyperedge_reprs = []
        node_updates = {}  # user_idx (local) -> list of contributions
        for edge in edges:
            member_idx = [user_index_map[u] for u in edge["users"] if u in user_index_map]
            if not member_idx:
                continue
            members = user_embeds[member_idx]                       # [k, embed_dim]
            mean_member = members.mean(dim=0)                       # [embed_dim]
            coord = torch.tensor(
                [edge["synchrony"], edge["burstiness"], edge["dispersion"]],
                dtype=torch.float, device=device,
            )
            h_edge = self.hyperedge_proj(torch.cat([mean_member, coord], dim=-1))  # [hidden]
            hyperedge_reprs.append(h_edge)

            keys = self.key_proj(h_edge).unsqueeze(0).expand(len(member_idx), -1)  # [k, hidden]
            values = self.value_proj(h_edge).unsqueeze(0).expand(len(member_idx), -1)
            queries = self.query_proj(members)                                     # [k, hidden]
            scores = (queries * keys).sum(dim=-1) / (keys.size(-1) ** 0.5)          # [k]
            weights = torch.softmax(scores, dim=0)
            contrib = weights.unsqueeze(-1) * values                                # [k, hidden]
            for local_i, uidx in enumerate(member_idx):
                node_updates.setdefault(uidx, []).append(contrib[local_i])

        if not node_updates:
            return torch.zeros(self.out_norm.normalized_shape[0], device=device)

        node_reprs = torch.stack([torch.stack(v).sum(dim=0) for v in node_updates.values()])
        node_reprs = self.dropout(self.out_norm(node_reprs))
        return node_reprs.mean(dim=0)  # aggregate over the cascade's nodes


class PropagationView(nn.Module):
    def __init__(self, user_vocab_size: int, user_embed_dim: int = 64, hidden_dim: int = 128,
                 output_dim: int = 256, num_layers: int = 2, dropout: float = 0.1,
                 enabled_relations: List[str] = None):
        super().__init__()
        self.user_embed = nn.Embedding(user_vocab_size, user_embed_dim, padding_idx=0)
        self.enabled_relations = enabled_relations or RELATIONS
        self.branches = nn.ModuleDict({
            rel: RelationBranch(user_embed_dim, hidden_dim, dropout) for rel in self.enabled_relations
        })
        self.num_layers = num_layers  # kept for config symmetry; branches are shallow by design
        self.relation_gate = nn.Sequential(
            nn.Linear(len(self.enabled_relations) * hidden_dim, len(self.enabled_relations)),
            nn.Softmax(dim=-1),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.hidden_dim = hidden_dim

    def _instance_forward(self, relations: Dict[str, List[dict]], historical_ctx: HistoricalContext,
                           device: torch.device) -> torch.Tensor:
        all_users = set()
        for rel in self.enabled_relations:
            for edge in relations.get(rel, []):
                all_users.update(edge["users"])
        if not all_users:
            all_users = {UNK_USER}
        all_users = sorted(all_users)
        user_index_map = {u: i for i, u in enumerate(all_users)}
        idx_tensor = torch.tensor(
            [historical_ctx.user_index(u) for u in all_users], dtype=torch.long, device=device,
        )
        user_embeds = self.user_embed(idx_tensor)

        branch_outputs = []
        for rel in self.enabled_relations:
            out = self.branches[rel](user_embeds, relations.get(rel, []), user_index_map, device)
            branch_outputs.append(out)
        stacked = torch.stack(branch_outputs)  # [n_relations, hidden]

        gate_input = stacked.flatten().unsqueeze(0)
        weights = self.relation_gate(gate_input).squeeze(0)  # [n_relations]
        combined = (weights.unsqueeze(-1) * stacked).sum(dim=0)
        return combined

    def forward(self, relations_batch: List[Dict[str, List[dict]]], historical_ctx: HistoricalContext,
                device: torch.device) -> torch.Tensor:
        outs = [self._instance_forward(r, historical_ctx, device) for r in relations_batch]
        return self.output_proj(torch.stack(outs))
