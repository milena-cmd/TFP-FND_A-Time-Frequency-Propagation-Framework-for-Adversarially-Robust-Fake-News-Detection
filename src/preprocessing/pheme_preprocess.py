"""PHEME preprocessing pipeline.

Implements the pipeline described in Section "Dataset Construction and Preprocessing":
  1. Walk all 9 events, both rumour/ and non-rumour/ thread folders.
  2. Keep only rumour threads with a veracity annotation of "true" or "false"
     (drop "unverified" and all non-rumour threads).
  3. Build the node set from the source tweet plus *every reaction JSON file
     physically present on disk* (not just whatever structure.json enumerates).
  4. Recover parent-child edges with the fallback order:
       (a) structure.json
       (b) in_reply_to_status_id on the reaction's own JSON
       (c) orphan (attached directly to the source node)
  5. Drop cascades with fewer than `min_cascade_events` total propagation events.
  6. Emit a per-instance JSONL manifest + an audit JSON mirroring the paper's
     "PHEME preprocessing audit" table so a re-run can be diffed against it.

Expected raw layout (official PHEME v1.0 release):
    <raw_dir>/<event>/{rumours,non-rumours}/<thread_id>/
        source-tweet/<thread_id>.json
        reactions/*.json
        structure.json
        annotation.json      (rumour threads only)
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

EVENTS = [
    "charliehebdo", "sydneysiege", "ferguson", "germanwings-crash",
    "ottawashooting", "putinmissing", "ebola-essien", "prince-toronto", "gurlitt",
]


def _to_epoch_seconds(raw) -> Optional[float]:
    """Normalize a PHEME timestamp to epoch seconds (float).

    PHEME tweet JSONs carry `timestamp_ms` in *milliseconds*; some entries only have
    the Twitter `created_at` string. Everything downstream (temporal windows, tau_seconds
    truncation, the 600s/300s/60s constants) assumes seconds, so this conversion must
    happen once, here, rather than being re-guessed in every consumer.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) or (isinstance(raw, str) and raw.strip().lstrip("-").isdigit()):
        value = float(raw)
        # timestamp_ms values are ~13 digits; plain unix seconds are ~10 digits.
        return value / 1000.0 if value > 1e12 else value
    if isinstance(raw, str):
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(raw).timestamp()
        except (TypeError, ValueError):
            return None
    return None


def _normalize_event_name(raw_name: str) -> str:
    """Canonicalize a PHEME event folder name to the keys used in
    src/preprocessing/splits.py's PHEME_SPLITS (e.g. "charliehebdo-all-rnr-threads"
    -> "charliehebdo"). The official all-rnr-annotated-threads release names its
    top-level folders with an "-all-rnr-threads" (or similar) suffix; without this
    normalization, split assignment silently fails to match any instances to any
    split, because it looks up instances by exact event-name equality.
    """
    name = raw_name.lower()
    for suffix in ("-all-rnr-annotated-threads", "-all-rnr-threads", "_all_rnr_threads", "-threads"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _read_json(path: Path) -> Optional[dict]:
    """Read one JSON file, tolerating the kinds of corruption real scraped-tweet
    archives actually contain (truncated downloads, mixed/invalid encodings) rather
    than letting one bad file among 100k+ crash the entire preprocessing run.

    json.JSONDecodeError is itself a ValueError subclass, and a mis-encoded byte
    sequence raises UnicodeDecodeError (also a ValueError subclass) when `open(...,
    encoding="utf-8")` reads it -- catching ValueError broadly covers both without
    needing to enumerate every possible decode failure by name.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        log.warning("Failed to parse %s (%s: %s)", path, type(exc).__name__, exc)
        return None


def _flag_value(raw) -> Optional[bool]:
    """Interpret a PHEME annotation flag (misinformation/true) as True/False/unknown.

    The raw archive mixes representations for the same semantic value across threads
    and even within the same event -- e.g. `"misinformation": 0` (int) on one thread
    and `"misinformation": "0"` (str) on another. Both must map to the same boolean.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("1", "true"):
            return True
        if s in ("0", "false"):
            return False
    return None


def _veracity_label(annotation: dict) -> Optional[int]:
    """Map a PHEME annotation.json to {1: true rumour, 0: false rumour, None: exclude}.

    Real schema (confirmed against the raw v1.0 release, not just documentation):
      - "misinformation" is present on every rumour thread; truthy -> confirmed FALSE,
        and in that case "true" is typically absent entirely.
      - When misinformation is falsy (or absent), "true" -- if present -- disambiguates
        confirmed TRUE (true: 1) from still-UNVERIFIED (true: 0). If "true" is also
        absent in that case, the thread is unverified too.
    Values may be encoded as either native ints/bools or numeric strings ("0"/"1"),
    inconsistently across events (esp. the 4 events added to make the 9-event
    veracity release) -- _flag_value normalizes both.
    """
    misinformation = _flag_value(annotation.get("misinformation"))
    if misinformation is True:
        return 0  # confirmed false rumour

    if "true" in annotation:
        true_flag = _flag_value(annotation.get("true"))
        if true_flag is True:
            return 1  # confirmed true rumour
        return None  # true: 0 (or unparseable) -> unverified

    return None  # no "true" key and not misinformation -> unverified


def _real_json_files(directory: Path) -> List[Path]:
    """List *.json files in a directory, excluding macOS AppleDouble metadata files
    (named "._<original filename>", created automatically whenever a folder is
    zipped/compressed on macOS -- they match the "*.json" glob but aren't valid JSON).
    Sorted for deterministic ordering, since a plain glob()'s order is filesystem-
    dependent and previously caused a junk file to sometimes be picked over the real
    one at index [0].
    """
    return sorted(p for p in directory.glob("*.json") if not p.name.startswith("._"))


def _load_thread(thread_dir: Path) -> Optional[dict]:
    # The distributed archive is inconsistent about singular vs. plural folder names
    # across mirrors/versions ("source-tweet" vs "source-tweets"); try both rather
    # than silently returning zero instances when the wrong one is hard-coded.
    source_dir = None
    for candidate in ("source-tweet", "source-tweets"):
        d = thread_dir / candidate
        if d.exists():
            source_dir = d
            break
    source_files = _real_json_files(source_dir) if source_dir is not None else []
    if not source_files:
        return None
    source = _read_json(source_files[0])
    if source is None:
        return None

    reactions_dir = None
    for candidate in ("reactions", "reaction"):
        d = thread_dir / candidate
        if d.exists():
            reactions_dir = d
            break
    reaction_files = _real_json_files(reactions_dir) if reactions_dir is not None else []
    reactions = {}
    for rf in reaction_files:
        r = _read_json(rf)
        if r is not None:
            reactions[str(r.get("id_str", rf.stem))] = r

    structure = _read_json(thread_dir / "structure.json") or {}

    return {"source": source, "reactions": reactions, "structure": structure}


def _flatten_structure(structure: dict) -> Dict[str, str]:
    """Flatten the nested structure.json dict into {child_id: parent_id}."""
    parent_of: Dict[str, str] = {}

    def _walk(node: dict, parent_id: Optional[str]):
        for key, children in node.items():
            if parent_id is not None:
                parent_of[key] = parent_id
            if isinstance(children, dict) and children:
                _walk(children, key)

    _walk(structure, None)
    return parent_of


def _build_cascade(thread: dict, source_id: str) -> List[dict]:
    """Return a list of propagation events: {id, user, ts, parent, depth}.

    `ts` is stored as *elapsed seconds since the source post* (source_ts subtracted
    here), matching what src/data/hypergraph.py and src/models/temporal_view.py both
    expect -- neither of those modules re-subtracts a source timestamp, so this
    conversion must happen exactly once, here.
    """
    source = thread["source"]
    reactions = thread["reactions"]
    structure_parent = _flatten_structure(thread["structure"])

    source_ts_abs = _to_epoch_seconds(source.get("timestamp_ms") or source.get("created_at")) or 0.0

    events = []
    id_to_ts = {source_id: 0.0}

    for rid, r in reactions.items():
        parent = structure_parent.get(rid)
        if parent is None:
            in_reply = r.get("in_reply_to_status_id_str")
            if in_reply and (in_reply == source_id or in_reply in reactions):
                parent = in_reply
        if parent is None:
            parent = source_id  # orphan -> attach directly to source

        abs_ts = _to_epoch_seconds(r.get("timestamp_ms") or r.get("created_at"))
        rel_ts = (abs_ts - source_ts_abs) if abs_ts is not None else None

        events.append({
            "id": rid,
            "user": (r.get("user") or {}).get("id_str", "unknown"),
            "ts": rel_ts,
            "parent": parent,
            "text": r.get("text", ""),
        })
        id_to_ts[rid] = events[-1]["ts"]

    # compute depth via parent chain (cap traversal to avoid cycles from bad data)
    def _depth(node_id: str, _hops: int = 0) -> int:
        if node_id == source_id or _hops > 64:
            return 0
        parent = next((e["parent"] for e in events if e["id"] == node_id), source_id)
        return 1 + _depth(parent, _hops + 1)

    for e in events:
        e["depth"] = _depth(e["id"])

    return events


def process_event(event_dir: Path) -> Tuple[List[dict], dict]:
    """Process a single event directory, return (instances, per-event audit counts)."""
    instances = []
    audit = {"non_rumour": 0, "rumour_true": 0, "rumour_false": 0, "rumour_unverified": 0,
              "labeled_but_unloadable": 0}

    non_rumour_dir = event_dir / "non-rumours"
    if non_rumour_dir.exists():
        audit["non_rumour"] = sum(1 for _ in non_rumour_dir.iterdir() if _.is_dir())

    rumour_dir = event_dir / "rumours"
    if not rumour_dir.exists():
        return instances, audit

    for thread_dir in sorted(p for p in rumour_dir.iterdir() if p.is_dir()):
        annotation = _read_json(thread_dir / "annotation.json")
        if annotation is None:
            continue
        label = _veracity_label(annotation)
        if label is None:
            audit["rumour_unverified"] += 1
            continue
        audit["rumour_true" if label == 1 else "rumour_false"] += 1

        thread = _load_thread(thread_dir)
        if thread is None:
            audit["labeled_but_unloadable"] += 1
            continue
        source_id = thread["source"].get("id_str", thread_dir.name)
        cascade = _build_cascade(thread, source_id)

        instances.append({
            "source_id": source_id,
            "event": _normalize_event_name(event_dir.name),
            "label": label,
            "text": thread["source"].get("text", ""),
            # cascade event "ts" values are already elapsed-seconds-since-source (see
            # _build_cascade), so source_ts is fixed at 0.0 here -- downstream code
            # (hypergraph.py, temporal_view.py) truncates/normalizes using
            # `ts - source_ts`, which must stay 0 for a cascade whose events are
            # already relative, or every horizon truncation double-subtracts.
            "source_ts": 0.0,
            "source_ts_absolute_unix": _to_epoch_seconds(
                thread["source"].get("timestamp_ms") or thread["source"].get("created_at")
            ),
            "cascade": cascade,
        })
    return instances, audit


def run(raw_dir: Path, out_dir: Path, min_cascade_events: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_instances: List[dict] = []
    audit_totals = {"non_rumour": 0, "rumour_true": 0, "rumour_false": 0, "rumour_unverified": 0,
                     "labeled_but_unloadable": 0}

    found_events = [d for d in raw_dir.iterdir() if d.is_dir()]
    if not found_events:
        raise FileNotFoundError(
            f"No event directories found under {raw_dir}. "
            f"Expected e.g. {raw_dir}/charliehebdo/rumours/..."
        )

    for event_dir in sorted(found_events):
        instances, audit = process_event(event_dir)
        all_instances.extend(instances)
        for k in audit_totals:
            audit_totals[k] += audit[k]
        log.info("Event %-20s | true=%d false=%d unverified=%d non_rumour=%d unloadable=%d",
                  event_dir.name, audit["rumour_true"], audit["rumour_false"],
                  audit["rumour_unverified"], audit["non_rumour"], audit["labeled_but_unloadable"])

    if audit_totals["labeled_but_unloadable"] > 0:
        log.warning(
            "%d threads had a valid, parseable annotation.json but their source-tweet/"
            "reactions content could not be loaded (missing/renamed folders or files) -- "
            "these are silently excluded from instances.jsonl. If this number is large, "
            "the raw archive's internal folder layout likely differs from what "
            "_load_thread() expects; inspect one such thread_dir's contents directly.",
            audit_totals["labeled_but_unloadable"],
        )

    total_true_false = audit_totals["rumour_true"] + audit_totals["rumour_false"]

    # cascade-length filter
    kept, removed_short = [], 0
    for inst in all_instances:
        total_events = 1 + len(inst["cascade"])  # +1 for the source post itself
        if total_events < min_cascade_events:
            removed_short += 1
            continue
        kept.append(inst)

    manifest_path = out_dir / "instances.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for inst in kept:
            f.write(json.dumps(inst) + "\n")

    n_true = sum(1 for i in kept if i["label"] == 1)
    n_false = sum(1 for i in kept if i["label"] == 0)

    audit_report = {
        "raw_events_found": len(found_events),
        "rumour_true_false_before_length_filter": total_true_false,
        "unverified_removed": audit_totals["rumour_unverified"],
        "labeled_but_unloadable": audit_totals["labeled_but_unloadable"],
        "cascade_lt_min_events_removed": removed_short,
        "min_cascade_events": min_cascade_events,
        "final_total": len(kept),
        "final_true": n_true,
        "final_false": n_false,
        "final_true_pct": round(100 * n_true / len(kept), 2) if kept else 0.0,
        "final_false_pct": round(100 * n_false / len(kept), 2) if kept else 0.0,
    }
    with open(out_dir / "preprocessing_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    log.info("Wrote %d instances to %s", len(kept), manifest_path)
    log.info("Audit: %s", json.dumps(audit_report, indent=2))
    log.info(
        "Compare final_total/final_true/final_false against the paper's Table "
        "\"Final PHEME dataset statistics after preprocessing\" (expects 1198/816/382 "
        "for the exact v1.0 release used in the paper -- differences usually mean a "
        "different PHEME release or a different min_cascade_events)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--min_cascade_events", type=int, default=5)
    args = parser.parse_args()
    run(args.raw_dir, args.out_dir, args.min_cascade_events)
