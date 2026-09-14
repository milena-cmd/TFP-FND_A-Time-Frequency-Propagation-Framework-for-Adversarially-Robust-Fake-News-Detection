"""Twitter15 / Twitter16 preprocessing.

Expects the standard Ma et al. (2016/2017) release layout:
    <raw_dir>/source_tweets.txt      # "<tweet_id>\t<text>"
    <raw_dir>/label.txt              # "<label>:<tweet_id>", label in
                                      #   {non-rumor, false, true, unverified}
    <raw_dir>/tree/<tweet_id>.txt    # one propagation edge per line:
                                      #   "['uid1', 'tid1', 't1']->['uid2', 'tid2', 't2']"
                                      #   the root edge has uid1 == 'ROOT'

Only "true" and "false" labelled source tweets are retained (conservative mapping,
Section "Twitter15 and Twitter16"). Propagation structures are reconstructed from the
tree/ files; records lacking enough metadata for a given relation type are excluded
from that relation rather than imputed (Section "Propagation reconstruction for
Twitter15 and Twitter16").
"""
import argparse
import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

LABEL_MAP = {"true": 1, "false": 0}  # non-rumor / unverified are excluded


def _load_labels(raw_dir: Path) -> Dict[str, int]:
    labels = {}
    with open(raw_dir / "label.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label_str, tid = line.split(":", 1)
            label_str = label_str.strip().lower()
            if label_str in LABEL_MAP:
                labels[tid.strip()] = LABEL_MAP[label_str]
    return labels


def _load_texts(raw_dir: Path) -> Dict[str, str]:
    texts = {}
    with open(raw_dir / "source_tweets.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            tid, text = line.split("\t", 1)
            texts[tid.strip()] = text
    return texts


def _parse_tree_line(line: str) -> Optional[tuple]:
    """Parse a "['u1','t1','ts1']->['u2','t2','ts2']" line into (parent, child)."""
    line = line.strip()
    if "->" not in line:
        return None
    left, right = line.split("->", 1)
    try:
        parent = ast.literal_eval(left.strip())
        child = ast.literal_eval(right.strip())
    except (ValueError, SyntaxError):
        return None
    return parent, child


def _build_cascade(tree_path: Path, source_id: str, time_unit_seconds: float = 60.0) -> List[dict]:
    """`time_unit_seconds` converts the tree file's raw elapsed-time field into seconds.

    The Ma et al. (2016/2017) release's third tuple element is an elapsed-time-since-
    source value, but the exact unit (minutes vs. seconds) has varied across mirrors of
    this dataset. The rest of the pipeline (window_seconds=600, early_window_seconds=300,
    sync_threshold_seconds=60, tau_seconds horizons) all assume seconds, matching
    `source_ts=0.0` for PHEME -- so verify this multiplier against a few known-good
    cascades in your specific release before trusting absolute-time-based features
    (temporal-cluster hyperedges, early-activity concentration, observation horizons).
    Default assumes minutes (time_unit_seconds=60.0); pass 1.0 if your release is
    already in seconds.
    """
    if not tree_path.exists():
        return []
    events, seen = [], set()
    with open(tree_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_tree_line(line)
            if parsed is None:
                continue
            (p_uid, p_tid, p_t), (c_uid, c_tid, c_t) = parsed
            if c_tid in seen or c_tid == source_id and p_uid == "ROOT":
                # the ROOT->source edge itself isn't a propagation event
                if p_uid == "ROOT":
                    continue
            if p_uid == "ROOT":
                continue
            seen.add(c_tid)
            try:
                t = float(c_t) * time_unit_seconds
            except (TypeError, ValueError):
                t = None
            events.append({
                "id": c_tid,
                "user": c_uid,
                "ts": t,           # elapsed seconds since source, after unit conversion
                "parent": p_tid,
                "depth": None,     # filled below
            })

    # depth via parent chain
    id_to_parent = {e["id"]: e["parent"] for e in events}

    def _depth(node_id: str, _hops: int = 0) -> int:
        if node_id == source_id or node_id not in id_to_parent or _hops > 128:
            return 0
        return 1 + _depth(id_to_parent[node_id], _hops + 1)

    for e in events:
        e["depth"] = _depth(e["id"])
    return events


def run(raw_dir: Path, dataset_name: str, out_dir: Path, time_unit_seconds: float = 60.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = _load_labels(raw_dir)
    texts = _load_texts(raw_dir)
    tree_dir = raw_dir / "tree"

    instances = []
    missing_tree = 0
    for tid, label in labels.items():
        if tid not in texts:
            continue
        cascade = _build_cascade(tree_dir / f"{tid}.txt", tid, time_unit_seconds)
        if not cascade:
            missing_tree += 1
        instances.append({
            "source_id": tid,
            "dataset": dataset_name,
            "label": label,
            "text": texts[tid],
            "source_ts": 0.0,   # Ma et al. release doesn't give absolute source timestamps
            "cascade": cascade,
        })

    manifest_path = out_dir / "instances.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")

    n_true = sum(1 for i in instances if i["label"] == 1)
    n_false = sum(1 for i in instances if i["label"] == 0)
    audit = {
        "dataset": dataset_name,
        "final_total": len(instances),
        "final_true": n_true,
        "final_false": n_false,
        "final_true_pct": round(100 * n_true / len(instances), 2) if instances else 0.0,
        "final_false_pct": round(100 * n_false / len(instances), 2) if instances else 0.0,
        "instances_with_empty_or_missing_tree_file": missing_tree,
    }
    with open(out_dir / "preprocessing_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    log.info("Wrote %d instances to %s", len(instances), manifest_path)
    log.info("Audit: %s", json.dumps(audit, indent=2))
    log.info(
        "Compare final_total/final_true/final_false against the paper's Table "
        "\"master_dataset_stats\" (expects 742/372/370 for Twitter15, 410/205/205 "
        "for Twitter16)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--dataset_name", choices=["twitter15", "twitter16"], required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--time_unit_seconds", type=float, default=60.0,
                         help="Multiplier converting the tree file's elapsed-time field to seconds "
                              "(default 60.0 assumes the field is in minutes; use 1.0 if already in seconds).")
    args = parser.parse_args()
    run(args.raw_dir, args.dataset_name, args.out_dir, args.time_unit_seconds)
