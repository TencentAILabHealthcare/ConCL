#!/bin/bash
DET_CFG=$1
WEIGHTS=$2
NUM_GPUS=$3
OUTPUT_DIR=$4
PY_ARGS=${@:5}



python $(dirname "$0")/train_net.py --config-file $DET_CFG \
    --num-gpus $NUM_GPUS \
    ${PY_ARGS} \
    MODEL.WEIGHTS $WEIGHTS\
    SOLVER.IMS_PER_BATCH $(($NUM_GPUS * 2)) \
    OUTPUT_DIR $OUTPUT_DIR