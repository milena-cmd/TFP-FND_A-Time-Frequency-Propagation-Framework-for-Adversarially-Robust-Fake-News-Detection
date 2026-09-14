"""Multi-relational hyperedge construction, Section "Hyperedge Construction".

Given a (possibly attacked) cascade truncated at a prediction deadline, builds four
relation types over the observed users:

  1. co_propagation  : one hyperedge with every user who propagated the item.
  2. temporal_cluster : non-overlapping `window_seconds`-wide windows; users whose
                         events fall in the same window form a hyperedge (windows
                         with <2 users are discarded).
  3. reply_thread     : for each reply node, one hyperedge = {node} U descendants up
                         to `max_depth`; hyperedges may overlap.
  4. community        : users sharing a historical community (passed in, computed
                         once per split from training-only interactions) form a
                         hyperedge.

Also computes the three coordination descriptors per hyperedge (Section "Hyperedge
Coordination Features"): synchrony, burstiness, temporal dispersion. Hyperedges with
fewer than two users get all three descriptors set to 0.0, matching the paper.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Hyperedge = List[str]  # list of user ids


def _event_ts(e: dict, source_ts: float) -> Optional[float]:
    ts = e.get("ts")
    if ts is None:
        return None
    try:
        return float(ts)
    except (TypeError, ValueError):
        return None


def truncate_prefix(cascade: List[dict], source_ts: float, tau_seconds: Optional[float]) -> List[dict]:
    """Strict prefix construction: keep only events with (ts - source_ts) <= tau."""
    if tau_seconds is None:
        return list(cascade)
    kept = []
    for e in cascade:
        ts = _event_ts(e, source_ts)
        if ts is None:
            continue
        if ts - source_ts <= tau_seconds:
            kept.append(e)
    return kept


def co_propagation_hyperedge(cascade: List[dict], source_user: str) -> List[Hyperedge]:
    users = {source_user} | {e.get("user", "unknown") for e in cascade}
    return [sorted(users)] if len(users) >= 1 else []


def temporal_cluster_hyperedges(cascade: List[dict], source_ts: float,
                                 window_seconds: float) -> List[Hyperedge]:
    if not cascade:
        return []
    timestamps = [(_event_ts(e, source_ts), e.get("user", "unknown")) for e in cascade]
    timestamps = [(t, u) for t, u in timestamps if t is not None]
    if not timestamps:
        return []
    t_min = min(t for t, _ in timestamps)
    buckets: Dict[int, set] = defaultdict(set)
    for t, u in timestamps:
        idx = int((t - t_min) // window_seconds)
        buckets[idx].add(u)
    return [sorted(users) for users in buckets.values() if len(users) >= 2]


def reply_thread_hyperedges(cascade: List[dict], source_id: str, max_depth: int) -> List[Hyperedge]:
    children: Dict[str, List[str]] = defaultdict(list)
    user_of = {source_id: source_id}
    for e in cascade:
        children[e["parent"]].append(e["id"])
        user_of[e["id"]] = e.get("user", "unknown")

    hyperedges = []
    all_ids = [source_id] + [e["id"] for e in cascade]
    for node_id in all_ids:
        collected = {user_of.get(node_id, "unknown")}
        frontier = [(node_id, 0)]
        while frontier:
            cur, depth = frontier.pop()
            if depth >= max_depth:
                continue
            for child in children.get(cur, []):
                collected.add(user_of.get(child, "unknown"))
                frontier.append((child, depth + 1))
        if len(collected) >= 2:
            hyperedges.append(sorted(collected))
    return hyperedges


def community_hyperedges(cascade: List[dict], source_user: str,
                          community_of: Dict[str, int]) -> List[Hyperedge]:
    users = {source_user} | {e.get("user", "unknown") for e in cascade}
    by_community: Dict[int, set] = defaultdict(set)
    for u in users:
        c = community_of.get(u)
        if c is not None:
            by_community[c].add(u)
    return [sorted(us) for us in by_community.values() if len(us) >= 2]


def coordination_features(hyperedge: Hyperedge, cascade: List[dict], source_ts: float,
                           sync_threshold_seconds: float = 60.0) -> Tuple[float, float, float]:
    """Returns (synchrony, burstiness, temporal_dispersion). Zero if <2 users."""
    if len(hyperedge) < 2:
        return 0.0, 0.0, 0.0

    user_set = set(hyperedge)
    ts_list = []
    for e in cascade:
        if e.get("user", "unknown") in user_set:
            t = _event_ts(e, source_ts)
            if t is not None:
                ts_list.append(t)
    if len(ts_list) < 2:
        return 0.0, 0.0, 0.0

    ts_arr = np.sort(np.asarray(ts_list))
    n_pairs = len(ts_arr) * (len(ts_arr) - 1) / 2
    sync_count = 0
    for i in range(len(ts_arr)):
        for j in range(i + 1, len(ts_arr)):
            if abs(ts_arr[i] - ts_arr[j]) <= sync_threshold_seconds:
                sync_count += 1
    synchrony = sync_count / n_pairs if n_pairs > 0 else 0.0

    window = 60.0
    bucket_idx = ((ts_arr - ts_arr.min()) // window).astype(int)
    _, counts = np.unique(bucket_idx, return_counts=True)
    burstiness = float(counts.max() / len(ts_arr))

    dispersion = float(np.var(ts_arr))

    return float(synchrony), burstiness, dispersion


def build_relations(cascade: List[dict], source_id: str, source_user: str, source_ts: float,
                     community_of: Dict[str, int], window_seconds: float = 600.0,
                     max_reply_depth: int = 5, sync_threshold_seconds: float = 60.0
                     ) -> Dict[str, List[Dict]]:
    """Returns {relation_name: [{"users": [...], "synchrony":.., "burstiness":.., "dispersion":..}]}."""
    relations = {
        "co_propagation": co_propagation_hyperedge(cascade, source_user),
        "temporal_cluster": temporal_cluster_hyperedges(cascade, source_ts, window_seconds),
        "reply_thread": reply_thread_hyperedges(cascade, source_id, max_reply_depth),
        "community": community_hyperedges(cascade, source_user, community_of),
    }
    out: Dict[str, List[Dict]] = {}
    for rel, edges in relations.items():
        out[rel] = []
        for edge in edges:
            sync, burst, disp = coordination_features(edge, cascade, source_ts, sync_threshold_seconds)
            out[rel].append({"users": edge, "synchrony": sync, "burstiness": burst, "dispersion": disp})
    return out
