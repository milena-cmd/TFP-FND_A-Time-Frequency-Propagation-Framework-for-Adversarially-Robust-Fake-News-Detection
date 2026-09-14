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

# Robustness-objective ablation: same tfp_fnd architecture, sweep lambda_robust/lambda_view.
declare -A ROBUST_CONFIGS=(
  ["classification_only"]="0.0 0.0"
  ["plus_robust"]="1.0 0.0"
  ["plus_view"]="0.0 0.5"
  ["full_objective"]="1.0 0.5"
)
for split in 1 2 3 4 5; do
  for name in "${!ROBUST_CONFIGS[@]}"; do
    read -r lr lv <<< "${ROBUST_CONFIGS[$name]}"
    for seed in "${SEEDS[@]}"; do
      echo ">>> [robustness] split=$split config=$name (lambda_r=$lr lambda_v=$lv) seed=$seed"
      python -m src.training.train --config "$CONFIG" --split "$split" --model tfp_fnd \
        --seed "$seed" --lambda_robust "$lr" --lambda_view "$lv"
    done
  done
done

python -m src.training.aggregate --results_dir results/pheme --models "${ARCH_ABLATIONS[@]}"
