#!/usr/bin/env bash
# RQ3 / Table "attack_families": generate each propagation-attack family, then train/eval
# every model against each one (kappa fixed at 0.10, matching the paper's main setting).
set -euo pipefail
PROCESSED_DIR=${1:-data/processed/pheme}
SPLIT_DIR="$PROCESSED_DIR/splits"
CONFIG=${2:-configs/pheme.yaml}
KAPPA=0.10
FAMILIES=(amplification temporal_camouflage structural_camouflage multi_community composite)

for family in "${FAMILIES[@]}"; do
  echo ">>> generating propagation attack family=$family"
  python -m src.attacks.propagation_attack --processed_dir "$PROCESSED_DIR" --split_dir "$SPLIT_DIR" \
    --kappa "$KAPPA" --family "$family" --seed 42
done

MODELS=(hgsl leht tfp_fnd_wo_robust tfp_fnd)
SEEDS=(0 1 2 3 4)
for split in 1 2 3 4 5; do
  for model in "${MODELS[@]}"; do
    for family in "${FAMILIES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo ">>> split=$split model=$model family=$family seed=$seed"
        python -m src.training.train --config "$CONFIG" --split "$split" --model "$model" \
          --seed "$seed" --attack_family "$family" --kappa "$KAPPA"
      done
    done
  done
done
