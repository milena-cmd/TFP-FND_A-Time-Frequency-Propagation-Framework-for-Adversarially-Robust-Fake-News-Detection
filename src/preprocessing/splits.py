"""Split generation.

PHEME: reproduces the exact event-disjoint assignment from the paper's Table
"Event-disjoint splits for PHEME" (5 splits, train/val/test events mutually exclusive
within each split -- Eqs. train-val / train-test / val-test disjointness).

Twitter15 / Twitter16: 5 stratified random splits (70/15/15 by default), preserving
each split's class balance, with a fixed seed so the split assignment is reproducible.

Both write out `<processed_dir>/splits/split_<k>.json` with
{"train": [source_ids], "val": [source_ids], "test": [source_ids]}, and every
attack-generation script downstream must assign a perturbed sample to the same
partition as its source item (never regenerate splits after attacks exist).
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Exact event assignment transcribed from the paper's Table "pheme_splits".
PHEME_SPLITS = [
    {
        "train": ["germanwings-crash", "putinmissing", "ebola-essien", "prince-toronto", "gurlitt"],
        "val": ["ferguson", "ottawashooting"],
        "test": ["charliehebdo", "sydneysiege"],
    },
    {
        "train": ["charliehebdo", "sydneysiege", "ebola-essien", "prince-toronto", "gurlitt"],
        "val": ["germanwings-crash", "putinmissing"],
        "test": ["ferguson", "ottawashooting"],
    },
    {
        "train": ["charliehebdo", "sydneysiege", "ferguson", "ottawashooting", "gurlitt"],
        "val": ["ebola-essien", "prince-toronto"],
        "test": ["germanwings-crash", "putinmissing"],
    },
    {
        "train": ["sydneysiege", "ferguson", "ottawashooting", "germanwings-crash", "putinmissing"],
        "val": ["gurlitt", "charliehebdo"],
        "test": ["ebola-essien", "prince-toronto"],
    },
    {
        "train": ["ottawashooting", "germanwings-crash", "putinmissing", "ebola-essien", "prince-toronto"],
        "val": ["sydneysiege", "ferguson"],
        "test": ["gurlitt", "charliehebdo"],
    },
]


def _load_instances(processed_dir: Path) -> List[dict]:
    path = processed_dir / "instances.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _assert_disjoint(split: Dict[str, List[str]]) -> None:
    train, val, test = set(split["train"]), set(split["val"]), set(split["test"])
    assert not (train & val), f"train/val overlap: {train & val}"
    assert not (train & test), f"train/test overlap: {train & test}"
    assert not (val & test), f"val/test overlap: {val & test}"


def build_pheme_splits(processed_dir: Path) -> None:
    instances = _load_instances(processed_dir)
    by_event: Dict[str, List[str]] = {}
    for inst in instances:
        by_event.setdefault(inst["event"], []).append(inst["source_id"])

    out_dir = processed_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, split_events in enumerate(PHEME_SPLITS, start=1):
        _assert_disjoint(split_events)
        split_ids = {
            part: [sid for ev in events for sid in by_event.get(ev, [])]
            for part, events in split_events.items()
        }
        missing_events = [
            ev for part in split_events.values() for ev in part if ev not in by_event
        ]
        if missing_events:
            log.warning(
                "Split %d references events with zero retained instances: %s "
                "(they may have been fully removed by preprocessing filters)",
                k, sorted(set(missing_events)),
            )
        with open(out_dir / f"split_{k}.json", "w", encoding="utf-8") as f:
            json.dump(split_ids, f, indent=2)
        log.info(
            "PHEME split %d: train=%d val=%d test=%d",
            k, len(split_ids["train"]), len(split_ids["val"]), len(split_ids["test"]),
        )


def build_stratified_splits(processed_dir: Path, num_splits: int = 5, seed: int = 42,
                             val_fraction_of_train: float = 0.15) -> None:
    instances = _load_instances(processed_dir)
    ids = [i["source_id"] for i in instances]
    labels = [i["label"] for i in instances]

    out_dir = processed_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    ids_arr, labels_arr = ids, labels
    for k, (train_val_idx, test_idx) in enumerate(skf.split(ids_arr, labels_arr), start=1):
        train_val_ids = [ids_arr[i] for i in train_val_idx]
        train_val_labels = [labels_arr[i] for i in train_val_idx]
        test_ids = [ids_arr[i] for i in test_idx]

        # carve a stratified validation slice out of train_val
        inner_skf = StratifiedKFold(
            n_splits=max(2, int(round(1 / val_fraction_of_train))),
            shuffle=True, random_state=seed + k,
        )
        tr_idx, val_idx = next(iter(inner_skf.split(train_val_ids, train_val_labels)))
        train_ids = [train_val_ids[i] for i in tr_idx]
        val_ids = [train_val_ids[i] for i in val_idx]

        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
        _assert_disjoint(split_ids)
        with open(out_dir / f"split_{k}.json", "w", encoding="utf-8") as f:
            json.dump(split_ids, f, indent=2)
        log.info(
            "Stratified split %d: train=%d val=%d test=%d",
            k, len(train_ids), len(val_ids), len(test_ids),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["pheme", "twitter15", "twitter16"], required=True)
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "pheme":
        build_pheme_splits(args.processed_dir)
    else:
        build_stratified_splits(args.processed_dir, num_splits=args.num_splits, seed=args.seed)
