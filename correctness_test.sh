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
OUTPUT_FILE="./build_test/output${STRATEGY}_${TEST_NUM}.txt"
ANSWER_FILE="./tests/correctness/testout${TEST_NUM}.txt"
TOL=6e-3

rm -rf ./build_test
mkdir ./build_test

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
MASS_FILE="./tests/correctness/testin${TEST_NUM}_mass.txt"
COORD_FILE="./tests/correctness/testin${TEST_NUM}_coordinate.txt"

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

if [ ! -f "$ANSWER_FILE" ]; then
    echo "ERROR: $ANSWER_FILE not found"
    exit 2
fi

awk -v tol="$TOL" '
function abs(x) { return x < 0 ? -x : x }

BEGIN { fail = 0 }

{
    if (NF < 3) {
        printf "ERROR: OUTPUT_FILE has <3 columns at line %d\n", NR
        fail = 1
        exit 2
    }

    if ((getline line < ans) <= 0) {
        printf "ERROR: ANSWER_FILE ended early at line %d\n", NR
        fail = 1
        exit 2
    }
    ans_lines++

    n = split(line, b)
    if (n < 3) {
        printf "ERROR: ANSWER_FILE has <3 columns at line %d\n", ans_lines
        fail = 1
        exit 2
    }

    for (i = 1; i <= 3; i++) {
        diff = abs($i - b[i])
        if (diff > tol) {
            printf "Mismatch line %d col %d: output=%.10g answer=%.10g |diff|=%.10g (tol=%g)\n",
                   NR, i, $i, b[i], diff, tol
            fail = 1
            exit 1
        }
    }
}

END {
    # If ANSWER_FILE has extra lines beyond what we consumed, detect it here
    if ((getline extra < ans) > 0 && fail == 0) {
        print "ERROR: ANSWER_FILE has extra lines"
        fail = 1
        exit 2
    }
    if (fail == 0){
        print "PASS"
    }
}
' ans="$ANSWER_FILE" "$OUTPUT_FILE"


echo "accuracy test of '${STRATEGY_NAME} version' using test case ${TEST_NUM} is done"
