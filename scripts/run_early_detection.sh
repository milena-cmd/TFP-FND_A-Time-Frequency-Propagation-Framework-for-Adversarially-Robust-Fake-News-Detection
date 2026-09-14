#!/usr/bin/env bash
# RQ5 / Table "early_deadline": requires a trained checkpoint per model/split/seed.
set -euo pipefail
CONFIG=${1:-configs/pheme.yaml}
MODELS=(bert_base bigcn hgsl leht tfp_fnd)
SEEDS=(0 1 2 3 4)

for split in 1 2 3 4 5; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      ckpt="checkpoints/pheme/${model}/split${split}_seed${seed}/best.pt"
      if [[ -f "$ckpt" ]]; then
        python -m src.training.evaluate_horizons --config "$CONFIG" --split "$split" \
          --model "$model" --checkpoint "$ckpt" --seed "$seed"
      else
        echo "SKIP: no checkpoint at $ckpt (run run_main_experiment.sh first)"
      fi
    done
  done
done
