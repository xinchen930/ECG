#!/bin/bash
# Run all schemes across 4 GPUs, each doing batch_1 → batch_2 → batch_1+2
#
# Usage:
#   bash scripts/run_all_parallel.sh              # default: a6000, random split
#   SERVER=3090 bash scripts/run_all_parallel.sh   # for 3090

set -e

SERVER=${SERVER:-a6000}
SPLIT=${SPLIT:-random}

echo "========================================"
echo "ECG Recon: Full Experiment Suite"
echo "Server: $SERVER | Split: $SPLIT"
echo "5 schemes x 3 batches = 15 training runs"
echo "========================================"

mkdir -p logs

# Step 0: Quality check for batch 2
QUALITY_CSV="eval_results/ppg_analysis_all_samples.csv"
if ! grep -q "pair_0098" "$QUALITY_CSV" 2>/dev/null; then
    echo "[Step 0] Running quality check for all samples..."
    python scripts/check_batch_quality.py 2>&1 | tee logs/quality_all.log
fi

# GPU 0: scheme_e (lightest) + scheme_i_direct
echo "[GPU 0] scheme_e + scheme_i_direct"
GPU=0 SERVER=$SERVER SPLIT=$SPLIT SCHEME=scheme_e bash scripts/run_all_batches.sh &
PID0=$!
# Wait for scheme_e to finish, then run scheme_i_direct on same GPU
wait $PID0
GPU=0 SERVER=$SERVER SPLIT=$SPLIT SCHEME=scheme_i_direct bash scripts/run_all_batches.sh &
PID0B=$!

# GPU 1: scheme_i_twostage
echo "[GPU 1] scheme_i_twostage"
GPU=1 SERVER=$SERVER SPLIT=$SPLIT SCHEME=scheme_i_twostage bash scripts/run_all_batches.sh &
PID1=$!

# GPU 2: scheme_f
echo "[GPU 2] scheme_f"
GPU=2 SERVER=$SERVER SPLIT=$SPLIT SCHEME=scheme_f bash scripts/run_all_batches.sh &
PID2=$!

# GPU 3: scheme_h (heaviest)
echo "[GPU 3] scheme_h"
GPU=3 SERVER=$SERVER SPLIT=$SPLIT SCHEME=scheme_h bash scripts/run_all_batches.sh &
PID3=$!

echo ""
echo "All experiments launched."
echo "Monitor: watch -n 5 nvidia-smi"
echo "Logs:    tail -f logs/scheme_*_b1.log"

# Wait for all
wait $PID0B $PID1 $PID2 $PID3

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "========================================"

# Collect results
echo ""
echo "Results summary:"
echo "----------------------------------------"
for scheme in scheme_e scheme_i_direct scheme_i_twostage scheme_f scheme_h; do
    echo ""
    echo "$scheme:"
    for tag in b1 b2 b1+2; do
        result=$(grep "Test results:" logs/${scheme}_${tag}.log 2>/dev/null | tail -1)
        printf "  %-6s %s\n" "$tag:" "${result:-'(not found)'}"
    done
done

echo ""
echo "Detailed JSON results:"
find checkpoints/ -name run_summary.json | sort | while read f; do
    echo "  $f"
done

echo ""
echo "To collect all results: find checkpoints/ -name run_summary.json -exec cat {} +"
