#!/usr/bin/env bash
# RQ2 / Table "ablation_arch" and RQ6 / Table "robustness_ablation".
set -euo pipefail
CONFIG=${1:-configs/pheme.yaml}
SEEDS=(0 1 2 3 4)

ARCH_ABLATIONS=(tfp_fnd tfp_fnd_wo_frequency tfp_fnd_wo_semantic tfp_fnd_wo_temporal \
                tfp_fnd_wo_hypergraph tfp_fnd_wo_adaptive_fusion tfp_fnd_pairwise_propagation)

for split in 1 2 3 4 5; do
  for model in "${ARCH_ABLATIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo ">>> [architectural] split=$split model=$model seed=$seed"
      python -m src.training.train --config "$CONFIG" --split "$split" --model "$model" --seed "$seed"
    done
  done
done

# Robustness-objective ablation (RQ6): same architecture, different training objectives.
#   classification_only          : clean instances only, L_cls
#   classification_plus_adv      : clean + adversarial variants, L_cls only (lambda_r = lambda_v = 0)
#   plus_robust                  : + L_robust
#   plus_view                    : + L_view
#   full_objective               : L_cls + lambda_r L_robust + lambda_v L_view
declare -A ROBUST_MODEL=(
  ["classification_only"]="tfp_fnd_clean_only"
  ["classification_plus_adv"]="tfp_fnd"
  ["plus_robust"]="tfp_fnd"
  ["plus_view"]="tfp_fnd"
  ["full_objective"]="tfp_fnd"
)
declare -A ROBUST_CONFIGS=(
  ["classification_only"]="0.0 0.0"
  ["classification_plus_adv"]="0.0 0.0"
  ["plus_robust"]="1.0 0.0"
  ["plus_view"]="0.0 0.5"
  ["full_objective"]="1.0 0.5"
)
for split in 1 2 3 4 5; do
  for name in "${!ROBUST_CONFIGS[@]}"; do
    read -r lr lv <<< "${ROBUST_CONFIGS[$name]}"
    for seed in "${SEEDS[@]}"; do
      echo ">>> [robustness] split=$split config=$name (lambda_r=$lr lambda_v=$lv) seed=$seed"
      python -m src.training.train --config "$CONFIG" --split "$split" --model "${ROBUST_MODEL[$name]}" \
        --seed "$seed" --lambda_robust "$lr" --lambda_view "$lv" --result_tag "$name"
    done
  done
done

python -m src.training.aggregate --results_dir results/pheme --models "${ARCH_ABLATIONS[@]}"
for name in "${!ROBUST_CONFIGS[@]}"; do
  python -m src.training.aggregate --results_dir results/pheme --models "${ROBUST_MODEL[$name]}" --result_tag "$name"
done
