#!/usr/bin/env bash
# RQ1 / Table "main_results": all baselines x 5 PHEME event-disjoint splits x 5 seeds.
set -euo pipefail
CONFIG=${1:-configs/pheme.yaml}
MODELS=(bert_base bert_dft bert_temporal bert_bigcn bert_hgsl bigcn hgsl leht tfp_fnd_wo_robust tfp_fnd)
SEEDS=(0 1 2 3 4)

for split in 1 2 3 4 5; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo ">>> split=$split model=$model seed=$seed"
      python -m src.training.train --config "$CONFIG" --split "$split" --model "$model" --seed "$seed"
    done
  done
done

python -m src.training.aggregate --results_dir results/pheme --models "${MODELS[@]}" \
  --compare tfp_fnd leht --condition joint
