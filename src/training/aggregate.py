"""Aggregate per-(split, seed) result JSONs into the mean+-std rows used in the paper's
tables. Follows the "Statistical Evaluation Protocol" section: seed-level results are
aggregated within a split first, then split-level values are aggregated to produce the
final mean+-std -- so a run with 5 splits x 5 seeds contributes 5 (not 25) independent
observations to the reported std.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from scipy import stats

from src.utils.metrics import mean_std


def load_results(results_dir: Path, model_name: str, result_tag: str = None) -> List[dict]:
    """result_tag filters to files matching the auto-generated attack-sweep suffix from
    src/training/train.py (e.g. "amplification_k0.1", "composite_k0.3") -- needed to
    aggregate run_attack_family_sweep.sh / run_attack_sensitivity.sh output, since those
    write `{model}_split{k}_seed{s}_{tag}.json` instead of the default
    `{model}_split{k}_seed{s}.json`.

    Matching is done with an anchored regex, not a glob + heuristic, because model names
    themselves contain underscores (e.g. "tfp_fnd", "tfp_fnd_wo_robust") -- any
    underscore-counting heuristic to tell a tagged filename from an untagged one breaks
    on those names.
    """
    escaped_model = re.escape(model_name)
    if result_tag:
        pattern = re.compile(rf"^{escaped_model}_split\d+_seed\d+_{re.escape(result_tag)}\.json$")
    else:
        pattern = re.compile(rf"^{escaped_model}_split\d+_seed\d+\.json$")

    records = []
    for path in results_dir.glob(f"{model_name}_split*_seed*.json"):
        if pattern.match(path.name):
            records.append(json.loads(path.read_text()))
    return records


def aggregate_model(results_dir: Path, model_name: str, result_tag: str = None) -> Dict:
    records = load_results(results_dir, model_name, result_tag)
    if not records:
        tag_msg = f" (result_tag={result_tag})" if result_tag else ""
        raise FileNotFoundError(f"No result files found for model={model_name}{tag_msg} in {results_dir}")

    by_split: Dict[int, List[dict]] = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)

    split_level_summaries = []
    for split_idx, seed_records in by_split.items():
        conditions = seed_records[0]["metrics_by_condition"].keys()
        split_summary = {"split": split_idx}
        for cond in conditions:
            f1s = [r["metrics_by_condition"][cond]["macro_f1"] for r in seed_records]
            split_summary[f"{cond}_macro_f1"] = sum(f1s) / len(f1s)  # seed-level aggregation
        split_level_summaries.append(split_summary)

    conditions = seed_records[0]["metrics_by_condition"].keys()
    final = {"model": model_name, "n_splits": len(split_level_summaries)}
    for cond in conditions:
        values = [s[f"{cond}_macro_f1"] for s in split_level_summaries]
        stat = mean_std(values)
        final[f"{cond}_macro_f1_mean"] = stat["mean"]
        final[f"{cond}_macro_f1_std"] = stat["std"]
    return final, split_level_summaries


def paired_significance(results_dir: Path, model_a: str, model_b: str, condition: str = "joint",
                          result_tag: str = None) -> Dict:
    """Paired t-test on split-level Macro-F1 between two models (Section
    "Statistical Evaluation Protocol"): the split, not the seed, is the paired unit."""
    _, splits_a = aggregate_model(results_dir, model_a, result_tag)
    _, splits_b = aggregate_model(results_dir, model_b, result_tag)
    a_by_split = {s["split"]: s[f"{condition}_macro_f1"] for s in splits_a}
    b_by_split = {s["split"]: s[f"{condition}_macro_f1"] for s in splits_b}
    common = sorted(set(a_by_split) & set(b_by_split))
    if len(common) < 2:
        return {"error": "fewer than 2 common splits; cannot run a paired test"}
    a_vals = [a_by_split[s] for s in common]
    b_vals = [b_by_split[s] for s in common]
    t_stat, p_value = stats.ttest_rel(a_vals, b_vals)
    return {
        "model_a": model_a, "model_b": model_b, "condition": condition,
        "n_paired_splits": len(common), "t_stat": float(t_stat), "p_value": float(p_value),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--compare", nargs=2, default=None, metavar=("MODEL_A", "MODEL_B"))
    parser.add_argument("--condition", default="joint")
    parser.add_argument("--result_tag", type=str, default=None,
                         help="Filter to a specific attack-sweep tag, e.g. 'amplification_k0.1' or "
                              "'composite_k0.3' (see the auto-generated suffix in "
                              "src/training/train.py::run). Omit for the default main-experiment results.")
    args = parser.parse_args()

    summaries = []
    for model in args.models:
        final, _ = aggregate_model(args.results_dir, model, args.result_tag)
        summaries.append(final)
        print(json.dumps(final, indent=2))

    tag_suffix = f"_{args.result_tag}" if args.result_tag else ""
    out_path = args.results_dir / f"aggregate_summary{tag_suffix}.json"
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {out_path}")

    if args.compare:
        sig = paired_significance(args.results_dir, args.compare[0], args.compare[1], args.condition,
                                    args.result_tag)
        print(json.dumps(sig, indent=2))
