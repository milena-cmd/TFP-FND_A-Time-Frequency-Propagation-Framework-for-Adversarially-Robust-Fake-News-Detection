"""RQ5 / Table "early_deadline": evaluate a trained checkpoint across observation
horizons. Rebuilds the test dataset once per horizon with the corresponding
tau_seconds so temporal/propagation relations are recomputed from the truncated
prefix only (Section "Strict prefix construction for early detection") -- the content
view is unaffected since source text is available at every horizon.

Example:
    python -m src.training.evaluate_horizons --config configs/pheme.yaml --split 1 \
        --model tfp_fnd --seed 0 \
        --checkpoint checkpoints/pheme/tfp_fnd/split1_seed0/best.pt
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.collate import make_collate_fn
from src.data.dataset import TFPFNDDataset
from src.data.historical import build_historical_context
from src.models.baselines import build_model
from src.models.tfp_fnd import TFPFNDConfig
from src.training.evaluate import evaluate_conditions
from src.training.train import load_config


def run(config_path: Path, split_idx: int, model_name: str, checkpoint: Path, seed: int) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = Path(cfg["dataset"]["processed_dir"])
    split_dir = Path(cfg["dataset"]["split_dir"])
    split = json.loads((split_dir / f"split_{split_idx}.json").read_text())
    with open(processed_dir / "instances.jsonl") as f:
        by_id = {json.loads(l)["source_id"]: json.loads(l) for l in f if l.strip()}
    train_instances = [by_id[i] for i in split["train"] if i in by_id]
    historical_ctx = build_historical_context(train_instances, resolution=cfg["temporal"]["community_resolution"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["bert_name"])
    collate_fn = make_collate_fn(tokenizer, max_seq_len=cfg["model"]["max_seq_len"])

    model_cfg = TFPFNDConfig(
        bert_name=cfg["model"]["bert_name"], bert_hidden=cfg["model"]["bert_hidden"],
        content_hidden=cfg["model"]["content_hidden"], temporal_layers=cfg["model"]["temporal_layers"],
        temporal_heads=cfg["model"]["temporal_heads"], temporal_hidden=cfg["model"]["temporal_hidden"],
        hypergraph_hidden=cfg["model"]["hypergraph_hidden"], propagation_hidden=cfg["model"]["propagation_hidden"],
        fusion_hidden=cfg["model"]["fusion_hidden"], dropout=cfg["model"]["dropout"],
        use_future_repost_count=cfg["temporal"]["use_future_repost_count"],
    )
    model = build_model(model_name, model_cfg, historical_ctx.vocab_size).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    results = {}
    for minutes in cfg["evaluation"]["observation_horizons_minutes"]:
        tau_seconds = minutes * 60.0
        dataset = TFPFNDDataset(
            processed_dir=processed_dir, split=split, partition="test",
            historical_ctx=historical_ctx, tau_seconds=tau_seconds,
            window_seconds=cfg["temporal"]["window_seconds"],
            max_reply_depth=cfg["temporal"]["reply_thread_max_depth"],
            sync_threshold_seconds=cfg["temporal"]["sync_threshold_seconds"],
            include_attacks=False,
        )
        loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False,
                             collate_fn=collate_fn)
        metrics = evaluate_conditions(model, loader, historical_ctx, tau_seconds, device,
                                       conditions=("clean",))
        results[f"{minutes}m"] = metrics["clean"].as_dict()
        print(f"horizon={minutes}m macro_f1={metrics['clean'].macro_f1:.4f}")

    out_dir = Path("results") / cfg["dataset"]["name"] / "early_detection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_split{split_idx}_seed{seed}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.config, args.split, args.model, args.checkpoint, args.seed)
