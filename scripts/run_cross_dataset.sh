#!/usr/bin/env bash
# Table "twitter_results": same models, Twitter15/Twitter16 configs.
set -euo pipefail
MODELS=(bert_base bigcn hgsl leht bert_bigcn bert_hgsl tfp_fnd)
SEEDS=(0 1 2 3 4)

for dataset in twitter15 twitter16; do
  for split in 1 2 3 4 5; do
    for model in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        python -m src.training.train --config "configs/${dataset}.yaml" --split "$split" \
          --model "$model" --seed "$seed"
      done
    done
  done
  python -m src.training.aggregate --results_dir "results/${dataset}" --models "${MODELS[@]}"
done
