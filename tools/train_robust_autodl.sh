#!/usr/bin/env bash
# Run with bash; CRLF-safe Git checkout is required on Linux as usual.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-v1}" in
  v1) EXPERIMENT=robust ;;
  v2) EXPERIMENT=robust_v2 ;;
  *) echo 'Usage: bash tools/train_robust_autodl.sh [v1|v2]' >&2; exit 1 ;;
esac
CONFIG="config/nturgbd-cross-subject/ctrgcn_mamba_${EXPERIMENT}_fast128.yaml"
WORK="work_dir/ntu60/xsub/ctrgcn_mamba_${EXPERIMENT}_joint_fast128"
TRAIN_LOG="logs/ntu60_xsub_joint_mamba_${EXPERIMENT}.log"
WATCH_LOG="logs/autodl_shutdown_mamba_${EXPERIMENT}.log"
if [[ -e "$WORK" || -e "$TRAIN_LOG" || -e "$WATCH_LOG" ]]; then
  echo 'Refusing to overwrite an existing robust run/log. Inspect or archive it first.' >&2
  exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
# Preflight only: this does not request shutdown.
python tools/autodl_shutdown.py --check-shutdown
python -c 'import numpy as np, torch, mamba_ssm, causal_conv1d; assert torch.cuda.is_available(); assert np.array_equal(torch.from_numpy(np.arange(3)).numpy(), np.arange(3)); print("Python/CUDA/NumPy/Mamba imports passed")'

nohup python -u main.py --config "$CONFIG" --device 0 --seed 1 \
  > "$TRAIN_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!
nohup python -u tools/autodl_shutdown.py \
  --pid "$TRAIN_PID" --config "$CONFIG" --work-dir "$WORK" \
  --training-log "$TRAIN_LOG" \
  > "$WATCH_LOG" 2>&1 < /dev/null &
WATCH_PID=$!
echo "Training PID: $TRAIN_PID; shutdown watcher PID: $WATCH_PID"
sleep 2
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
  tail -n 40 "$TRAIN_LOG"
  echo 'Training exited during startup. Inspect the log; do not launch a duplicate run.' >&2
  exit 1
fi
if ! kill -0 "$WATCH_PID" 2>/dev/null; then
  tail -n 40 "$WATCH_LOG"
  echo 'Shutdown watcher exited; training may still be running. Inspect the watcher log.' >&2
  exit 1
fi
tail -n 10 "$WATCH_LOG"
echo "Training log: $TRAIN_LOG"
echo "Shutdown log: $WATCH_LOG (must show Armed)"
