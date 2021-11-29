#!/bin/bash
DET_CFG=$1
NUM_GPUS=$2
OUTPUT_DIR=$3
PY_ARGS=${@:4}


# train detection model from scratch
python $(dirname "$0")/train_net.py --config-file $DET_CFG \
    --num-gpus $NUM_GPUS \
    ${PY_ARGS} \
    SOLVER.IMS_PER_BATCH $(($NUM_GPUS * 2)) \
    OUTPUT_DIR $OUTPUT_DIR