"""Historical statistics, Section "Historical User and Propagation Representation".

Everything here is computed ONCE per split from `train`-partition interactions only,
excluding the target cascade, and is treated as a fixed lookup table at train/val/test
time (Section "Historical Information Protocol"). This is what prevents inductive
leakage: a user's embedding index and community id are only ever assigned if the user
appeared during training; unseen users fall back to a shared UNK id.
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx

try:
    import community as community_louvain
except ImportError:  # pragma: no cover
    community_louvain = None

UNK_USER = "<UNK_USER>"


class HistoricalContext:
    def __init__(self, user_to_idx: Dict[str, int], community_of: Dict[str, int],
                 user_activity: Dict[str, int], user_hyperedges: Dict[str, List[dict]]):
        self.user_to_idx = user_to_idx
        self.community_of = community_of
        self.user_activity = user_activity
        self.user_hyperedges = user_hyperedges  # co-propagation + community only, per Section

    def user_index(self, user_id: str) -> int:
        return self.user_to_idx.get(user_id, self.user_to_idx[UNK_USER])

    @property
    def vocab_size(self) -> int:
        return len(self.user_to_idx)

    def save(self, path: Path) -> None:
        payload = {
            "user_to_idx": self.user_to_idx,
            "community_of": self.community_of,
            "user_activity": self.user_activity,
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "HistoricalContext":
        payload = json.loads(path.read_text())
        return cls(payload["user_to_idx"], payload["community_of"], payload["user_activity"], {})


def build_historical_context(train_instances: List[dict], resolution: float = 1.0) -> HistoricalContext:
    g = nx.Graph()
    activity = defaultdict(int)
    co_prop_membership: Dict[str, List[dict]] = defaultdict(list)

    for inst in train_instances:
        author = inst.get("author", inst["source_id"])  # PHEME/Twitter releases rarely expose a clean author field
        users = {author}
        for e in inst["cascade"]:
            u = e.get("user", "unknown")
            users.add(u)
            activity[u] += 1
        activity[author] += 1
        users = list(users)
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                if g.has_edge(users[i], users[j]):
                    g[users[i]][users[j]]["weight"] += 1
                else:
                    g.add_edge(users[i], users[j], weight=1)

    if community_louvain is not None and g.number_of_nodes() > 0:
        community_of = community_louvain.best_partition(g, resolution=resolution, random_state=0)
    else:
        community_of = {n: 0 for n in g.nodes}

    for inst in train_instances:
        author = inst.get("author", inst["source_id"])
        users = {author} | {e.get("user", "unknown") for e in inst["cascade"]}
        by_community: Dict[int, List[str]] = defaultdict(list)
        for u in users:
            c = community_of.get(u)
            if c is not None:
                by_community[c].append(u)
        for c, us in by_community.items():
            if len(us) >= 2:
                for u in us:
                    co_prop_membership[u].append({"users": sorted(us), "relation": "community"})
        if len(users) >= 2:
            for u in users:
                co_prop_membership[u].append({"users": sorted(users), "relation": "co_propagation"})

    all_users = sorted(set(g.nodes) | set(activity.keys()))
    user_to_idx = {UNK_USER: 0}
    for u in all_users:
        user_to_idx[u] = len(user_to_idx)

    return HistoricalContext(user_to_idx, dict(community_of), dict(activity), dict(co_prop_membership))
