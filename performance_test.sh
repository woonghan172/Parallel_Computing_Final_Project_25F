#!/bin/bash
set -euo pipefail

# -------- Argument Parsing --------
if [ $# -lt 2 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 [strategy] [test-case-number]"
    echo "Strategies:"
    echo "  0: serial"
    echo "  1: shared memory"
    echo "  2: thread coasening"
    echo "  3: optimal"
    echo "  4: hierarchical"
    echo "  5: cutlass"
    exit 1
fi

STRATEGY=$1
TEST_NUM=$2
OUTPUT_FILE="./build_perf_test/output${STRATEGY}_${TEST_NUM}.txt"

rm -rf ./build_perf_test
mkdir ./build_perf_test

# -------- Strategy Selection --------
case $STRATEGY in
    0)
        EXEC="./build/serial"
        STRATEGY_NAME="serial"
        ;;
    1)
        EXEC="./build/shared_memory"
        STRATEGY_NAME="shared_memory"
        ;;
    2)
        EXEC="./build/thread_coarsening"
        STRATEGY_NAME="thread_coarsening"
        ;;
    3)
        EXEC="./build/optimal"
        STRATEGY_NAME="optimal"
        ;;
    4)
        EXEC="./build/hierarchical"
        STRATEGY_NAME="hierarchical"
        ;;
    5)
        EXEC="./build/cutlass"
        STRATEGY_NAME="cutlass"
        ;;
    *)
        echo "[ERROR] Invalid strategy: $STRATEGY"
        exit 1
        ;;
esac

# -------- Input Test Files --------
MASS_FILE="./tests/speedup/s_testin${TEST_NUM}_mass.txt"
COORD_FILE="./tests/speedup/s_testin${TEST_NUM}_coordinate.txt"

if [ ! -f "$MASS_FILE" ]; then
    echo "[ERROR] Mass file not found: $MASS_FILE"
    exit 1
fi

if [ ! -f "$COORD_FILE" ]; then
    echo "[ERROR] Coordinate file not found: $COORD_FILE"
    exit 1
fi

$EXEC "$MASS_FILE" "$COORD_FILE" "$OUTPUT_FILE"

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "ERROR: $OUTPUT_FILE not found"
    exit 2
fi

#echo "performance test of '${STRATEGY_NAME} version' using test case ${TEST_NUM} is done"
