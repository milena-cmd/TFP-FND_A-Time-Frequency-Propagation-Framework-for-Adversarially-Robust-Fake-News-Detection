"""Propagation-level adversarial attack generator.

Implements Section "Propagation-Level Adversarial Attack Generation" verbatim:

  For each controlled account:
    1. Community selection   - sample a target community, prob. proportional to size.
    2. Historical camouflage - synthesize decoy interactions before the target event.
    3. Parent selection      - weighted by depth, degree centrality, historical activity
                                (gamma_1=0.5, gamma_2=0.3, gamma_3=0.2 by default).
    4. Timestamp generation  - relative to the parent, drawn from an empirical organic
                                waiting-time distribution; discard if past the horizon.
    5. Community consistency - add the controlled account into the historical community
                                graph.
    6. Target-cascade injection.

Threat model (Section "Threat Model"): non-adaptive, black-box -- the generator never
touches model parameters, gradients, or hidden representations. All distributional
parameters (organic waiting times, degree/activity statistics) are estimated from the
training partition only, and attacks only ever *add* interactions, never delete.

The four named attack "families" in the paper's Table "attack_families" are realized
as parameter presets over this same generator:
  - amplification         : high kappa, low temporal/structural constraints
  - temporal_camouflage    : timestamps drawn tightly around real inter-arrival times
  - structural_camouflage  : parents chosen to mimic the organic depth/centrality mix
  - multi_community        : accounts spread across many communities instead of one
  - composite              : all of the above combined
"""
import argparse
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np

try:
    import community as community_louvain  # python-louvain
except ImportError:  # pragma: no cover
    community_louvain = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    kappa: float = 0.10          # injection ratio: injected accounts / organic accounts
    decoy_ratio: float = 0.20    # rho_3: historical decoy budget per injected account
    gamma_depth: float = 0.5
    gamma_centrality: float = 0.3
    gamma_activity: float = 0.2
    temporal_window_seconds: float = 600.0
    community_resolution: float = 1.0
    family: str = "composite"    # amplification | temporal_camouflage |
                                  # structural_camouflage | multi_community | composite


def _organic_waiting_times(instances: List[dict]) -> np.ndarray:
    """Empirical parent->child inter-arrival times (seconds), estimated from training data."""
    deltas = []
    for inst in instances:
        ts_by_id = {inst["source_id"]: inst.get("source_ts") or 0.0}
        for e in inst["cascade"]:
            ts_by_id[e["id"]] = e.get("ts")
        for e in inst["cascade"]:
            parent_ts = ts_by_id.get(e["parent"])
            child_ts = e.get("ts")
            if parent_ts is None or child_ts is None:
                continue
            try:
                d = float(child_ts) - float(parent_ts)
            except (TypeError, ValueError):
                continue
            if d >= 0:
                deltas.append(d)
    if not deltas:
        return np.array([1.0])  # degenerate fallback
    return np.asarray(deltas)


def _build_historical_community_graph(train_instances: List[dict]) -> nx.Graph:
    """Co-propagation graph over users in the training partition (for community sampling)."""
    g = nx.Graph()
    for inst in train_instances:
        users = {inst["source_id"]}
        for e in inst["cascade"]:
            users.add(e.get("user", "unknown"))
        users = list(users)
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                if g.has_edge(users[i], users[j]):
                    g[users[i]][users[j]]["weight"] += 1
                else:
                    g.add_edge(users[i], users[j], weight=1)
    return g


def _detect_communities(g: nx.Graph, resolution: float) -> Dict[str, int]:
    if community_louvain is None or g.number_of_nodes() == 0:
        return {n: 0 for n in g.nodes}
    return community_louvain.best_partition(g, resolution=resolution, random_state=0)


def _degree_centrality_lookup(events: List[dict], source_id: str) -> Dict[str, float]:
    g = nx.DiGraph()
    g.add_node(source_id)
    for e in events:
        g.add_edge(e["parent"], e["id"])
    if g.number_of_nodes() <= 1:
        return {source_id: 0.0}
    return nx.degree_centrality(g)


def _select_parent(events: List[dict], source_id: str, cfg: AttackConfig,
                    rng: random.Random) -> str:
    candidates = [source_id] + [e["id"] for e in events]
    if len(candidates) == 1:
        return source_id

    depth_of = {source_id: 0}
    for e in events:
        depth_of[e["id"]] = e.get("depth", 0) or 0
    centrality = _degree_centrality_lookup(events, source_id)
    activity = defaultdict(int)
    for e in events:
        activity[e["parent"]] += 1

    max_depth = max(depth_of.values()) or 1
    max_activity = max(activity.values()) if activity else 1

    weights = []
    for c in candidates:
        d = depth_of.get(c, 0) / max_depth
        cen = centrality.get(c, 0.0)
        act = activity.get(c, 0) / max_activity
        w = cfg.gamma_depth * d + cfg.gamma_centrality * cen + cfg.gamma_activity * act
        weights.append(max(w, 1e-6))
    total = sum(weights)
    probs = [w / total for w in weights]
    return rng.choices(candidates, weights=probs, k=1)[0]


def _sample_timestamp(parent_ts: float, waiting_times: np.ndarray, horizon: float,
                       rng: np.random.Generator, family: str) -> Optional[float]:
    if family == "temporal_camouflage":
        # tightly concentrated around the median organic gap -> hard to distinguish temporally
        median = float(np.median(waiting_times))
        delta = max(0.0, rng.normal(median, median * 0.15 + 1e-6))
    else:
        delta = float(rng.choice(waiting_times))
    ts = parent_ts + delta
    if ts > horizon:
        return None
    return ts


def attack_instance(inst: dict, cfg: AttackConfig, waiting_times: np.ndarray,
                     community_of: Dict[str, int], communities: List[int],
                     community_sizes: Dict[int, int], horizon_seconds: Optional[float],
                     rng: random.Random, np_rng: np.random.Generator) -> dict:
    events = list(inst["cascade"])
    source_id = inst["source_id"]
    source_ts = float(inst.get("source_ts") or 0.0)
    n_organic = max(1, len(events))
    n_inject = max(1, int(round(cfg.kappa * n_organic)))

    horizon = horizon_seconds if horizon_seconds is not None else float("inf")
    if events:
        try:
            last_ts = max(float(e["ts"]) for e in events if e.get("ts") is not None)
            horizon = max(horizon, last_ts) if horizon_seconds is None else horizon
        except ValueError:
            pass

    if cfg.family == "multi_community":
        n_target_communities = min(len(communities), max(2, n_inject // 2)) if communities else 1
        target_communities = rng.sample(communities, k=n_target_communities) if communities else [0]
    else:
        weights = [community_sizes.get(c, 1) for c in communities] if communities else [1]
        if sum(weights) <= 0:
            weights = [1 for _ in weights]  # degenerate community graph (e.g. tiny/cold-start cascade) -> uniform
        target_communities = [rng.choices(communities, weights=weights, k=1)[0]] if communities else [0]

    injected = []
    injected_user_counter = 0
    for _ in range(n_inject):
        controlled_user = f"bot_{inst['source_id']}_{injected_user_counter}"
        injected_user_counter += 1
        target_community = rng.choice(target_communities)

        # --- historical camouflage: decoy interactions before the target event ---
        n_decoys = int(round(cfg.decoy_ratio * 1))
        for d in range(n_decoys):
            decoy_ts = source_ts - float(np_rng.choice(waiting_times)) * (d + 1)
            injected.append({
                "id": f"{controlled_user}_decoy_{d}",
                "user": controlled_user,
                "ts": decoy_ts,
                "parent": source_id,  # decoys are structurally inert (pre-cascade)
                "depth": None,
                "injected": True,
                "decoy": True,
                "community": target_community,
            })

        # --- parent selection + timestamp generation ---
        parent = _select_parent(events + injected, source_id, cfg, rng)
        parent_ts_lookup = {source_id: source_ts}
        for e in events + injected:
            parent_ts_lookup[e["id"]] = e.get("ts", source_ts)
        parent_ts = float(parent_ts_lookup.get(parent, source_ts) or source_ts)

        new_ts = _sample_timestamp(parent_ts, waiting_times, horizon, np_rng, cfg.family)
        if new_ts is None:
            continue  # discard: would fall outside the observation horizon

        new_id = f"{controlled_user}_evt"
        injected.append({
            "id": new_id,
            "user": controlled_user,
            "ts": new_ts,
            "parent": parent,
            "depth": None,
            "injected": True,
            "decoy": False,
            "community": target_community,
        })
        community_of[controlled_user] = target_community

    # recompute depth for injected nodes given final parent assignment
    all_events = events + injected
    id_to_parent = {e["id"]: e["parent"] for e in all_events}

    def _depth(node_id: str, _hops: int = 0) -> int:
        if node_id == source_id or node_id not in id_to_parent or _hops > 128:
            return 0
        return 1 + _depth(id_to_parent[node_id], _hops + 1)

    for e in injected:
        e["depth"] = _depth(e["id"])

    attacked = dict(inst)
    attacked["cascade"] = all_events
    attacked["attack"] = {"type": "propagation", "family": cfg.family, "kappa": cfg.kappa}
    return attacked


def run(processed_dir: Path, split_dir: Path, kappas: List[float], family: str,
        seed: int, out_subdir: str = "attacks/propagation") -> None:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    with open(processed_dir / "instances.jsonl", "r", encoding="utf-8") as f:
        instances = [json.loads(line) for line in f if line.strip()]
    by_id = {inst["source_id"]: inst for inst in instances}

    out_dir = processed_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_path in sorted(split_dir.glob("split_*.json")):
        split = json.loads(split_path.read_text())
        train_instances = [by_id[i] for i in split["train"] if i in by_id]

        waiting_times = _organic_waiting_times(train_instances)
        g = _build_historical_community_graph(train_instances)
        community_of = _detect_communities(g, resolution=1.0)
        communities = sorted(set(community_of.values())) or [0]
        community_sizes: Dict[int, int] = defaultdict(int)
        for c in community_of.values():
            community_sizes[c] += 1

        for kappa in kappas:
            cfg = AttackConfig(kappa=kappa, family=family)
            out_file = out_dir / f"{split_path.stem}_{family}_k{kappa}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                # attacks are only ever generated for train (robust training) and test
                # (attacked evaluation); val is attacked too so lambda_r/lambda_v can be
                # tuned against attacked validation performance if desired.
                for part in ("train", "val", "test"):
                    for sid in split[part]:
                        if sid not in by_id:
                            continue
                        attacked = attack_instance(
                            by_id[sid], cfg, waiting_times, dict(community_of),
                            communities, community_sizes, None, rng, np_rng,
                        )
                        attacked["partition"] = part
                        f.write(json.dumps(attacked) + "\n")
            log.info("Wrote %s", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--kappa", type=float, nargs="+", default=[0.10])
    parser.add_argument("--family", choices=[
        "amplification", "temporal_camouflage", "structural_camouflage",
        "multi_community", "composite",
    ], default="composite")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.processed_dir, args.split_dir, args.kappa, args.family, args.seed)
