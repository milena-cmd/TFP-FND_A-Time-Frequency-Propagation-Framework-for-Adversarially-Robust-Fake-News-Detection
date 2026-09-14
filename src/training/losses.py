"""Joint Robust Optimization losses, Section "Joint Robust Optimization".

L_cls    (Eq. classification_loss) : mean BCE over the 4 conditions {clean, c, p, cp}.
L_robust (Eq. prediction_consistency) : Jensen-Shannon divergence between clean and
          each attacked condition's predicted distribution.
L_view   : cosine-distance consistency per the invariance structure in
          Table "invariance" -- content attacks should leave the temporal/propagation
          views unchanged, propagation attacks should leave the content view unchanged,
          joint attacks are not penalized on any view (all may legitimately change).

Both L_robust and L_view are masked per-sample by has_content_attack /
has_propagation_attack (from the collate function) so that samples lacking a real
attacked variant (clean fallback) don't spuriously count as "should be identical"
comparisons -- since content==clean or cascade==clean already trivially, they'd
contribute ~0 loss anyway, but masking keeps the reported loss interpretable.
"""
from typing import Dict

import torch
import torch.nn.functional as F

# view index convention inside v_prime: 0=content, 1=temporal, 2=propagation
CONTENT_VIEW_IDX, TEMPORAL_VIEW_IDX, PROPAGATION_VIEW_IDX = 0, 1, 2


def classification_loss(logits_by_condition: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    losses = [F.cross_entropy(logits_by_condition[cond], labels) for cond in ("clean", "content", "propagation", "joint")]
    return torch.stack(losses).mean()


def _js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.clamp_min(1e-8).log(), p, reduction="none").sum(-1)
    kl_qm = F.kl_div(m.clamp_min(1e-8).log(), q, reduction="none").sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def prediction_consistency_loss(logits_by_condition: Dict[str, torch.Tensor],
                                 has_content_attack: torch.Tensor,
                                 has_propagation_attack: torch.Tensor) -> torch.Tensor:
    clean = logits_by_condition["clean"]
    js_content = _js_divergence(clean, logits_by_condition["content"])
    js_prop = _js_divergence(clean, logits_by_condition["propagation"])
    js_joint = _js_divergence(clean, logits_by_condition["joint"])

    has_joint_attack = has_content_attack | has_propagation_attack
    terms, weights = [], []
    for js, mask in ((js_content, has_content_attack), (js_prop, has_propagation_attack),
                      (js_joint, has_joint_attack)):
        if mask.any():
            terms.append(js[mask].mean())
            weights.append(mask.float().mean())
    if not terms:
        return torch.tensor(0.0, device=clean.device)
    return torch.stack(terms).mean()


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def view_consistency_loss(v_prime_by_condition: Dict[str, torch.Tensor],
                           has_content_attack: torch.Tensor,
                           has_propagation_attack: torch.Tensor) -> torch.Tensor:
    """Table "invariance": content attack should leave temporal/propagation unchanged;
    propagation attack should leave content unchanged; joint attack is unconstrained."""
    clean = v_prime_by_condition["clean"]
    content = v_prime_by_condition["content"]
    propagation = v_prime_by_condition["propagation"]

    n_views = clean.size(1)
    terms = []

    if n_views > TEMPORAL_VIEW_IDX and has_content_attack.any():
        d = _cosine_distance(clean[:, TEMPORAL_VIEW_IDX], content[:, TEMPORAL_VIEW_IDX])
        terms.append(d[has_content_attack].mean())
    if n_views > PROPAGATION_VIEW_IDX and has_content_attack.any():
        d = _cosine_distance(clean[:, PROPAGATION_VIEW_IDX], content[:, PROPAGATION_VIEW_IDX])
        terms.append(d[has_content_attack].mean())
    if has_propagation_attack.any():
        d = _cosine_distance(clean[:, CONTENT_VIEW_IDX], propagation[:, CONTENT_VIEW_IDX])
        terms.append(d[has_propagation_attack].mean())

    if not terms:
        return torch.tensor(0.0, device=clean.device)
    return torch.stack(terms).mean()


def total_loss(outputs_by_condition: Dict[str, Dict[str, torch.Tensor]], labels: torch.Tensor,
                has_content_attack: torch.Tensor, has_propagation_attack: torch.Tensor,
                lambda_robust: float, lambda_view: float) -> Dict[str, torch.Tensor]:
    logits_by_condition = {c: o["logits"] for c, o in outputs_by_condition.items()}
    v_prime_by_condition = {c: o["v_prime"] for c, o in outputs_by_condition.items()}

    l_cls = classification_loss(logits_by_condition, labels)
    l_robust = prediction_consistency_loss(logits_by_condition, has_content_attack, has_propagation_attack)
    l_view = view_consistency_loss(v_prime_by_condition, has_content_attack, has_propagation_attack)

    l_total = l_cls + lambda_robust * l_robust + lambda_view * l_view
    return {"total": l_total, "cls": l_cls, "robust": l_robust, "view": l_view}
