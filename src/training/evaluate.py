"""Evaluation loop.

`evaluate_conditions` runs the model over a dataloader once per requested condition
(clean/content/propagation/joint) and returns ClassificationMetrics for each --
directly reusable for Table "main_results" (all four), Table "twitter_results"
(clean + joint), and Table "attack_families" (clean + a specific propagation-attack
dataloader passed in as "propagation").
"""
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.data.historical import HistoricalContext
from src.utils.metrics import ClassificationMetrics, attack_degradation, compute_metrics, retention_ratio


@torch.no_grad()
def evaluate_conditions(model: torch.nn.Module, dataloader: DataLoader, historical_ctx: HistoricalContext,
                         tau_seconds: float, device: torch.device,
                         conditions: List[str] = ("clean", "content", "propagation", "joint"),
                         ) -> Dict[str, ClassificationMetrics]:
    model.eval()
    preds_by_condition = {c: [] for c in conditions}
    labels_all = []

    for batch in dataloader:
        labels_all.extend(batch["label"].tolist())
        for cond in conditions:
            out = model(batch["conditions"][cond], historical_ctx, tau_seconds, device)
            preds = out["logits"].argmax(dim=-1).cpu().tolist()
            preds_by_condition[cond].extend(preds)

    return {cond: compute_metrics(labels_all, preds_by_condition[cond]) for cond in conditions}


def summarize_with_robustness(metrics_by_condition: Dict[str, ClassificationMetrics]) -> Dict[str, float]:
    clean_f1 = metrics_by_condition["clean"].macro_f1
    joint_f1 = metrics_by_condition.get("joint", metrics_by_condition["clean"]).macro_f1
    return {
        "clean_macro_f1": clean_f1,
        "joint_macro_f1": joint_f1,
        "delta_joint": attack_degradation(clean_f1, joint_f1),
        "retention_ratio": retention_ratio(clean_f1, joint_f1),
    }
