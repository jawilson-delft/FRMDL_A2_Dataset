#!/usr/bin/env bash
# Sequential lambda_kink sweep: 175, 350, 700 (lambda=100 baseline already exists).
set -euo pipefail
cd "$(dirname "$0")"
PY="../fourier_neural_operator/.venv/bin/python"
for LAM in 175 350 700; do
  LABEL="lambda_${LAM}"
  LOG="results/training/${LABEL}/train.log"
  mkdir -p "results/training/${LABEL}" "checkpoints/${LABEL}"
  echo "=== Starting lambda_kink=${LAM} ($(date)) ===" | tee -a "$LOG"
  "$PY" -u train.py --epochs 200 --lambda-kink "$LAM" --run-label "$LABEL" 2>&1 | tee -a "$LOG"
  echo "=== Finished lambda_kink=${LAM} ($(date)) ===" | tee -a "$LOG"
done
echo "=== Running comparison ===" | tee -a results/training/lambda_sweep.log
"$PY" -u compare_lambda_sweep.py 2>&1 | tee -a results/training/lambda_sweep.log
