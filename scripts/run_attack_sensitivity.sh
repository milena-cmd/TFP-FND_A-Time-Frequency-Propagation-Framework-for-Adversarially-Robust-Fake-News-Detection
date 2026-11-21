#!/usr/bin/env bash
# Table "attack_sensitivity": sweep kappa (decoy_ratio/window/gamma_depth require editing
# configs/pheme.yaml's attacks.propagation block and re-running propagation_attack.py,
# since those parameters live in AttackConfig defaults rather than the CLI).
set -euo pipefail
PROCESSED_DIR=${1:-data/processed/pheme}
SPLIT_DIR="$PROCESSED_DIR/splits"
CONFIG=${2:-configs/pheme.yaml}
KAPPAS=(0.10 0.20 0.30 0.50)

for kappa in "${KAPPAS[@]}"; do
  python -m src.attacks.propagation_attack --processed_dir "$PROCESSED_DIR" --split_dir "$SPLIT_DIR" \
    --kappa "$kappa" --family composite --seed 42
done

for split in 1 2 3 4 5; do
  for kappa in "${KAPPAS[@]}"; do
    echo ">>> split=$split kappa=$kappa"
    python -m src.training.train --config "$CONFIG" --split "$split" --model tfp_fnd \
      --seed 0 --attack_family composite --kappa "$kappa"
  done
done

echo "Results land in results/pheme/tfp_fnd_split<k>_seed0_composite_k<kappa>.json"
echo "Diff each kappa's joint-attack macro_f1 against the kappa=0.10 (or your chosen"
echo "reference) run's clean macro_f1 to populate Table \"attack_sensitivity\"'s Delta column."
