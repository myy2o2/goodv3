#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRIALS="${TRIALS:-50}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-./outputs_baseline}"
EPOCHS="${EPOCHS:-50}"

declare -a RUNS=(
  "citeseer degree concept"
  "citeseer word covariate"
  "cora degree concept"
  "cora word covariate"
)

for spec in "${RUNS[@]}"; do
  read -r dataset domain shift <<<"$spec"
  ckpt="./outputs/stage1/${dataset}/${domain}/${shift}/right/pretrain_model.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt" >&2
    exit 1
  fi

  echo "== Running eerm_right on ${dataset}/${domain}/${shift} =="
  python baseline/run_baseline_ttt.py \
    --method eerm_right \
    --pretrain-ckpt "$ckpt" \
    --dataset "$dataset" \
    --domain "$domain" \
    --shift "$shift" \
    --finetune-epochs "$EPOCHS" \
    --use-optuna \
    --optuna-trials "$TRIALS" \
    --device "$DEVICE" \
    --output-root "$OUT_ROOT" \
    --timestamp "eerm_right_t${TRIALS}" \
    --run-name "baseline_compare_eerm_right"
done
