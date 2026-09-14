"""Train a single (config, split, model, seed) run.

Example:
    python -m src.training.train --config configs/pheme.yaml --split 1 \
        --model tfp_fnd --seed 0

Writes:
    checkpoints/<dataset>/<model>/split<k>_seed<s>/best.pt
    results/<dataset>/<model>_split<k>_seed<s>.json   (per-condition test metrics)
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.collate import make_collate_fn
from src.data.dataset import TFPFNDDataset
from src.data.historical import HistoricalContext, build_historical_context
from src.models.baselines import build_model
from src.models.tfp_fnd import TFPFNDConfig
from src.training.evaluate import evaluate_conditions, summarize_with_robustness
from src.training.losses import total_loss
from src.utils.seed import set_seed


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_datasets(cfg: dict, split_idx: int, tokenizer, historical_ctx: HistoricalContext,
                    attack_family: str = "composite", kappa: float = None):
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    split_dir = Path(cfg["dataset"]["split_dir"])
    split = json.loads((split_dir / f"split_{split_idx}.json").read_text())

    content_cache = processed_dir / "attack_cache" / "content_attacks.jsonl"
    kappa = kappa if kappa is not None else cfg["attacks"]["propagation"]["kappa_default"]
    prop_attack_file = processed_dir / "attacks" / "propagation" / \
        f"split_{split_idx}_{attack_family}_k{kappa}.jsonl"

    datasets = {}
    for partition in ("train", "val", "test"):
        datasets[partition] = TFPFNDDataset(
            processed_dir=processed_dir, split=split, partition=partition,
            historical_ctx=historical_ctx,
            tau_seconds=None,  # use full observed prefix at train/eval-clean time
            window_seconds=cfg["temporal"]["window_seconds"],
            max_reply_depth=cfg["temporal"]["reply_thread_max_depth"],
            sync_threshold_seconds=cfg["temporal"]["sync_threshold_seconds"],
            content_attack_cache=content_cache if content_cache.exists() else None,
            propagation_attack_file=prop_attack_file if prop_attack_file.exists() else None,
            include_attacks=(partition != "val" or True),
        )
    return datasets


def build_optimizer(model: torch.nn.Module, cfg: dict) -> AdamW:
    bert_params, head_params = [], []
    for name, param in model.named_parameters():
        if "bert" in name:
            bert_params.append(param)
        else:
            head_params.append(param)
    return AdamW([
        {"params": bert_params, "lr": cfg["training"]["lr_bert"]},
        {"params": head_params, "lr": cfg["training"]["lr_head"]},
    ], weight_decay=cfg["training"]["weight_decay"])


def run(config_path: Path, split_idx: int, model_name: str, seed: int,
        lambda_robust_override=None, lambda_view_override=None,
        attack_family: str = "composite", kappa: float = None,
        result_tag: str = None, force: bool = False) -> None:
    cfg = load_config(config_path)

    # Resume-safety: skip (model, split, seed, attack_tag) combos whose result JSON
    # already exists. This matters a lot in practice for Colab-style sessions where a
    # multi-hour sweep script can be interrupted and re-launched -- without this check,
    # every rerun silently redoes finished work (or worse, overwrites a good checkpoint
    # with a fresh one if training is non-deterministic across environments).
    default_kappa_for_tag = cfg["attacks"]["propagation"]["kappa_default"]
    tag_for_check = result_tag or f"{attack_family}_k{kappa if kappa is not None else default_kappa_for_tag}"
    is_default_attack_config_for_check = (attack_family == "composite" and
                                           (kappa is None or kappa == default_kappa_for_tag))
    results_dir_for_check = Path("results") / cfg["dataset"]["name"]
    fname_for_check = f"{model_name}_split{split_idx}_seed{seed}.json" if is_default_attack_config_for_check \
        else f"{model_name}_split{split_idx}_seed{seed}_{tag_for_check}.json"
    existing_result_path = results_dir_for_check / fname_for_check
    if existing_result_path.exists() and not force:
        print(f"SKIP: {existing_result_path} already exists (pass --force to retrain and overwrite it).")
        return

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = Path(cfg["dataset"]["processed_dir"])
    split_dir = Path(cfg["dataset"]["split_dir"])
    split = json.loads((split_dir / f"split_{split_idx}.json").read_text())
    with open(processed_dir / "instances.jsonl") as f:
        by_id = {json.loads(l)["source_id"]: json.loads(l) for l in f if l.strip()}
    train_instances = [by_id[i] for i in split["train"] if i in by_id]

    historical_ctx = build_historical_context(
        train_instances, resolution=cfg["temporal"]["community_resolution"],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_name"])
    collate_fn = make_collate_fn(tokenizer, max_seq_len=cfg["model"]["max_seq_len"])

    datasets = build_datasets(cfg, split_idx, tokenizer, historical_ctx,
                               attack_family=attack_family, kappa=kappa)
    loaders = {
        part: DataLoader(
            ds, batch_size=cfg["training"]["batch_size"], shuffle=(part == "train"),
            collate_fn=collate_fn,
        )
        for part, ds in datasets.items()
    }

    model_cfg = TFPFNDConfig(
        bert_name=cfg["model"]["bert_name"], bert_hidden=cfg["model"]["bert_hidden"],
        content_hidden=cfg["model"]["content_hidden"], temporal_layers=cfg["model"]["temporal_layers"],
        temporal_heads=cfg["model"]["temporal_heads"], temporal_hidden=cfg["model"]["temporal_hidden"],
        hypergraph_hidden=cfg["model"]["hypergraph_hidden"], propagation_hidden=cfg["model"]["propagation_hidden"],
        fusion_hidden=cfg["model"]["fusion_hidden"], dropout=cfg["model"]["dropout"],
        use_future_repost_count=cfg["temporal"]["use_future_repost_count"],
    )
    model = build_model(model_name, model_cfg, historical_ctx.vocab_size).to(device)
    optimizer = build_optimizer(model, cfg)

    lambda_robust = lambda_robust_override if lambda_robust_override is not None else cfg["training"]["lambda_robust"]
    lambda_view = lambda_view_override if lambda_view_override is not None else cfg["training"]["lambda_view"]
    if model_name == "tfp_fnd_wo_robust":
        lambda_robust, lambda_view = 0.0, 0.0

    best_val_f1, patience_left = -1.0, cfg["training"]["early_stopping_patience"]
    default_kappa = cfg["attacks"]["propagation"]["kappa_default"]
    tag = result_tag or f"{attack_family}_k{kappa if kappa is not None else default_kappa}"
    is_default_attack_config = (attack_family == "composite" and (kappa is None or kappa == default_kappa))
    ckpt_subdir = f"split{split_idx}_seed{seed}" if is_default_attack_config else f"split{split_idx}_seed{seed}_{tag}"
    ckpt_dir = Path("checkpoints") / cfg["dataset"]["name"] / model_name / ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["training"]["max_epochs"]):
        model.train()
        running = {"total": 0.0, "cls": 0.0, "robust": 0.0, "view": 0.0}
        n_batches = 0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            outputs_by_condition = {
                cond: model(batch["conditions"][cond], historical_ctx, tau_seconds=float("inf"), device=device)
                for cond in ("clean", "content", "propagation", "joint")
            }
            losses = total_loss(
                outputs_by_condition, batch["label"].to(device),
                batch["has_content_attack"].to(device), batch["has_propagation_attack"].to(device),
                lambda_robust, lambda_view,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            optimizer.step()

            for k in running:
                running[k] += float(losses[k].detach().cpu())
            n_batches += 1

        avg = {k: v / max(1, n_batches) for k, v in running.items()}
        val_metrics = evaluate_conditions(
            model, loaders["val"], historical_ctx, tau_seconds=float("inf"), device=device,
            conditions=("clean",),
        )
        val_f1 = val_metrics["clean"].macro_f1
        print(f"[epoch {epoch}] train_loss={avg['total']:.4f} "
              f"(cls={avg['cls']:.4f} robust={avg['robust']:.4f} view={avg['view']:.4f}) "
              f"val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_left = cfg["training"]["early_stopping_patience"]
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best val macro_f1={best_val_f1:.4f})")
                break

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    test_metrics = evaluate_conditions(
        model, loaders["test"], historical_ctx, tau_seconds=float("inf"), device=device,
    )
    summary = summarize_with_robustness(test_metrics)

    results_dir = Path("results") / cfg["dataset"]["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    result_record = {
        "dataset": cfg["dataset"]["name"], "model": model_name, "split": split_idx, "seed": seed,
        "lambda_robust": lambda_robust, "lambda_view": lambda_view,
        "attack_family": attack_family, "kappa": kappa if kappa is not None else default_kappa,
        "metrics_by_condition": {c: m.as_dict() for c, m in test_metrics.items()},
        "summary": summary,
    }
    fname = f"{model_name}_split{split_idx}_seed{seed}.json" if is_default_attack_config \
        else f"{model_name}_split{split_idx}_seed{seed}_{tag}.json"
    out_path = results_dir / fname
    out_path.write_text(json.dumps(result_record, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda_robust", type=float, default=None)
    parser.add_argument("--lambda_view", type=float, default=None)
    parser.add_argument("--attack_family", type=str, default="composite",
                         choices=["amplification", "temporal_camouflage", "structural_camouflage",
                                  "multi_community", "composite"],
                         help="Which propagation-attack file to train/evaluate the p/cp conditions "
                              "against (Table 'attack_families'); must already exist under "
                              "<processed_dir>/attacks/propagation/, see src/attacks/propagation_attack.py.")
    parser.add_argument("--kappa", type=float, default=None,
                         help="Injection ratio of the propagation-attack file to use "
                              "(Table 'attack_sensitivity'); defaults to configs/*.yaml's kappa_default.")
    parser.add_argument("--result_tag", type=str, default=None,
                         help="Override the auto-generated result/checkpoint suffix.")
    parser.add_argument("--force", action="store_true",
                         help="Retrain even if a result JSON for this (model, split, seed, attack "
                              "config) already exists, overwriting it.")
    args = parser.parse_args()
    run(args.config, args.split, args.model, args.seed, args.lambda_robust, args.lambda_view,
        args.attack_family, args.kappa, args.result_tag, args.force)
