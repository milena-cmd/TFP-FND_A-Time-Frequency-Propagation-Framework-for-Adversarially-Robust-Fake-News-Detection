"""Aggregate RQ5 / Table "early_deadline" results.

Reads every results/<dataset>/early_detection/{model}_split{k}_seed{s}.json written by
src/training/evaluate_horizons.py (one file per model/split/seed, each holding
Macro-F1 -- among other metrics -- at every configured observation horizon), and
produces mean+-std Macro-F1 per horizon, aggregated the same way as
src/training/aggregate.py: seeds are averaged within a split first, then the resulting
per-split values are aggregated into a final mean+-std, so a run with 5 splits x 5
seeds contributes 5 independent observations, not 25.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.utils.metrics import mean_std


def load_horizon_results(early_detection_dir: Path, model_name: str) -> List[dict]:
    pattern = re.compile(rf"^{re.escape(model_name)}_split(\d+)_seed(\d+)\.json$")
    records = []
    for path in early_detection_dir.glob(f"{model_name}_split*_seed*.json"):
        m = pattern.match(path.name)
        if not m:
            continue
        split_idx, seed = int(m.group(1)), int(m.group(2))
        payload = json.loads(path.read_text())
        records.append({"split": split_idx, "seed": seed, "horizons": payload})
    return records


def aggregate_horizons(early_detection_dir: Path, model_name: str) -> Dict:
    records = load_horizon_results(early_detection_dir, model_name)
    if not records:
        raise FileNotFoundError(
            f"No early-detection result files found for model={model_name} in {early_detection_dir}"
        )

    horizon_labels = sorted(
        records[0]["horizons"].keys(),
        key=lambda h: float(h.rstrip("m")),
    )

    by_split: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)

    split_level: Dict[str, List[float]] = {h: [] for h in horizon_labels}
    for split_idx, seed_records in by_split.items():
        for h in horizon_labels:
            seed_values = [r["horizons"][h]["macro_f1"] for r in seed_records if h in r["horizons"]]
            if seed_values:
                split_level[h].append(sum(seed_values) / len(seed_values))  # seed-level aggregation

    final = {"model": model_name, "n_splits": len(by_split)}
    for h in horizon_labels:
        stat = mean_std(split_level[h])
        final[f"{h}_macro_f1_mean"] = stat["mean"]
        final[f"{h}_macro_f1_std"] = stat["std"]
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True,
                         help="The dataset's results dir, e.g. results/pheme "
                              "(early_detection/ is expected directly under it).")
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    early_detection_dir = args.results_dir / "early_detection"
    summaries = []
    for model in args.models:
        final = aggregate_horizons(early_detection_dir, model)
        summaries.append(final)
        print(json.dumps(final, indent=2))

    out_path = early_detection_dir / "aggregate_summary.json"
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {out_path}")
