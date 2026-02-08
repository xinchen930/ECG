#!/bin/bash
# Run all schemes on Batch 1 data across 4 A6000 GPUs
# Usage: bash scripts/run_batch1.sh [SERVER_TYPE]
#   SERVER_TYPE: a6000 (default) or 3090

set -e
SERVER=${1:-a6000}
TAG="batch1"
SPLIT="random"

echo "========================================"
echo "ECG Recon Batch 1 Experiments"
echo "Server: $SERVER | Split: $SPLIT | Tag: $TAG"
echo "========================================"

# Create log directory
mkdir -p logs

# GPU 0: Scheme E (lightest) → Scheme I-Direct (sequential on same GPU)
echo "[GPU 0] Starting Scheme E + I-Direct..."
CUDA_VISIBLE_DEVICES=0 bash -c "
  python models/train.py --config configs/scheme_e.yaml --split $SPLIT --server $SERVER --run-tag $TAG 2>&1 | tee logs/scheme_e_${TAG}.log
  echo '--- Scheme E done, starting I-Direct ---'
  python models/train.py --config configs/scheme_i_direct.yaml --split $SPLIT --server $SERVER --run-tag $TAG 2>&1 | tee logs/scheme_i_direct_${TAG}.log
" &
PID0=$!

# GPU 1: Scheme I-TwoStage
echo "[GPU 1] Starting Scheme I-TwoStage..."
CUDA_VISIBLE_DEVICES=1 python models/train.py \
  --config configs/scheme_i_twostage.yaml --split $SPLIT --server $SERVER --run-tag $TAG \
  2>&1 | tee logs/scheme_i_twostage_${TAG}.log &
PID1=$!

# GPU 2: Scheme F (EfficientPhys)
echo "[GPU 2] Starting Scheme F..."
CUDA_VISIBLE_DEVICES=2 python models/train.py \
  --config configs/scheme_f.yaml --split $SPLIT --server $SERVER --run-tag $TAG \
  2>&1 | tee logs/scheme_f_${TAG}.log &
PID2=$!

# GPU 3: Scheme H (PhysFormer, heaviest)
echo "[GPU 3] Starting Scheme H..."
CUDA_VISIBLE_DEVICES=3 python models/train.py \
  --config configs/scheme_h.yaml --split $SPLIT --server $SERVER --run-tag $TAG \
  2>&1 | tee logs/scheme_h_${TAG}.log &
PID3=$!

echo ""
echo "All experiments launched. PIDs: GPU0=$PID0 GPU1=$PID1 GPU2=$PID2 GPU3=$PID3"
echo "Monitor: watch -n 5 nvidia-smi"
echo "Logs:    tail -f logs/scheme_*_${TAG}.log"
echo ""

# Wait for all to finish
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"

# Print summary
echo ""
echo "Results summary:"
echo "----------------------------------------"
for f in logs/scheme_*_${TAG}.log; do
  scheme=$(basename "$f" | sed "s/_${TAG}.log//")
  result=$(grep "Test results:" "$f" 2>/dev/null | tail -1)
  if [ -n "$result" ]; then
    echo "  $scheme: $result"
  else
    echo "  $scheme: (no result found - check log)"
  fi
done

echo ""
echo "Detailed results in: checkpoints/*/run_summary.json"
echo "Collect all: find checkpoints/ -name run_summary.json -exec cat {} +"
