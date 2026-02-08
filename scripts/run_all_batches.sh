#!/bin/bash
# Sequential training: batch_1 → batch_2 → batch_1+2
# Runs one scheme at a time on the specified GPU
#
# Usage:
#   GPU=0 SCHEME=scheme_e bash scripts/run_all_batches.sh
#   GPU=1 SCHEME=scheme_f bash scripts/run_all_batches.sh
#
# Or use run_all_parallel.sh to run all schemes across 4 GPUs.

set -e

GPU=${GPU:-0}
SCHEME=${SCHEME:-scheme_e}
SERVER=${SERVER:-a6000}
SPLIT=${SPLIT:-random}

echo "========================================"
echo "Training $SCHEME on GPU $GPU"
echo "Server: $SERVER | Split: $SPLIT"
echo "========================================"

mkdir -p logs

# Step 0: Quality check for batch 2 (only run once)
QUALITY_CSV="eval_results/ppg_analysis_all_samples.csv"
if ! grep -q "pair_0098" "$QUALITY_CSV" 2>/dev/null; then
    echo "[Step 0] Running quality check for batch 2..."
    CUDA_VISIBLE_DEVICES=$GPU python scripts/check_batch_quality.py --batch batch_2 2>&1 | tee logs/quality_batch2.log
    echo "[Step 0] Quality check done."
fi

# Step 1: Batch 1 only (98 samples from batch_1)
echo ""
echo "[Step 1/3] Training on batch_1 (98 samples)..."
CUDA_VISIBLE_DEVICES=$GPU python models/train.py \
    --config configs/${SCHEME}.yaml \
    --split $SPLIT --server $SERVER \
    --batch batch_1 \
    --run-tag b1 \
    2>&1 | tee logs/${SCHEME}_b1.log

# Step 2: Batch 2 only (37 samples from batch_2)
echo ""
echo "[Step 2/3] Training on batch_2 (37 samples)..."
CUDA_VISIBLE_DEVICES=$GPU python models/train.py \
    --config configs/${SCHEME}.yaml \
    --split $SPLIT --server $SERVER \
    --batch batch_2 \
    --run-tag b2 \
    --quality-filter all \
    2>&1 | tee logs/${SCHEME}_b2.log

# Step 3: Batch 1+2 combined (135 samples)
echo ""
echo "[Step 3/3] Training on batch_1+2 (135 samples)..."
CUDA_VISIBLE_DEVICES=$GPU python models/train.py \
    --config configs/${SCHEME}.yaml \
    --split $SPLIT --server $SERVER \
    --batch batch_1,batch_2 \
    --run-tag b1+2 \
    2>&1 | tee logs/${SCHEME}_b1+2.log

echo ""
echo "========================================"
echo "$SCHEME: All 3 batches completed!"
echo "========================================"

# Print results summary
echo ""
echo "Results:"
for tag in b1 b2 b1+2; do
    result=$(grep "Test results:" logs/${SCHEME}_${tag}.log 2>/dev/null | tail -1)
    echo "  $tag: ${result:-'(not found)'}"
done
