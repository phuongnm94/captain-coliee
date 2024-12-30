#!/bin/bash
ROOT_DIR=./

CONFIG=(monot5-large-10k_hns)

N_TRIALS=( 5 10 )
TMP_DIR=tmp
VAL_SEGMENT=val

export PYTHONPATH=$PWD

mkdir -p ./train_logs/tuned

SCRIPT=src/hp_search.py

for i in ${!CONFIG[@]}
do
    CONFIG_PATH=./configs/tuning/${CONFIG[$i]}.json
    python -u $SCRIPT $CONFIG_PATH ${N_TRIALS[$i]} $TMP_DIR/${CONFIG[$i]} \
        $VAL_SEGMENT $ROOT_DIR | tee ./train_logs/tuned/${CONFIG[$i]}.log
done
