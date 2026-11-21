#!/usr/bin/env bash
# Run this immediately after mounting Drive and symlinking data/processed, checkpoints,
# and results -- BEFORE launching any training sweep. It writes a small marker file
# through each symlink and then re-reads it via the *direct* Drive path, so a broken or
# missing symlink (e.g. because results/checkpoints were plain local directories left
# over from a previous session, and `ln -s` silently failed) is caught in seconds
# instead of after hours of "successful" training that never actually persisted anywhere.
set -euo pipefail

WORKSPACE=${1:-/content/drive/MyDrive/tfp-fnd-workspace}
FAIL=0

check_symlink() {
  local link_path=$1
  local expected_target=$2
  if [[ ! -L "$link_path" ]]; then
    echo "FAIL: $link_path is not a symlink at all (it's a plain file/directory, or missing)."
    FAIL=1
    return
  fi
  local actual_target
  actual_target=$(readlink -f "$link_path")
  local expected_resolved
  expected_resolved=$(readlink -f "$expected_target")
  if [[ "$actual_target" != "$expected_resolved" ]]; then
    echo "FAIL: $link_path points to $actual_target, expected $expected_resolved"
    FAIL=1
    return
  fi
  local marker="$link_path/.persistence_check_$$"
  echo "ok" > "$marker" 2>/dev/null || { echo "FAIL: could not write through $link_path"; FAIL=1; return; }
  local marker_via_drive="$expected_target/.persistence_check_$$"
  if [[ -f "$marker_via_drive" ]]; then
    echo "OK: $link_path -> $expected_resolved (write-through confirmed)"
    rm -f "$marker"
  else
    echo "FAIL: wrote through $link_path but it did not appear at $marker_via_drive"
    FAIL=1
  fi
}

check_symlink "data/processed" "$WORKSPACE/data_processed"
check_symlink "checkpoints"    "$WORKSPACE/checkpoints"
check_symlink "results"        "$WORKSPACE/results"

if [[ "$FAIL" -eq 1 ]]; then
  echo
  echo "=== Persistence check FAILED. Do not start training until this is fixed. ==="
  echo "Likely fix: rm -rf data/processed checkpoints results, then recreate the three"
  echo "symlinks pointing at $WORKSPACE/{data_processed,checkpoints,results}."
  exit 1
else
  echo
  echo "=== Persistence check PASSED. Safe to start training. ==="
fi
