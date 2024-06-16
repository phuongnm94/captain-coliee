#!/bin/bash

ROOT_DIR=/home/s2210405/codes/coliee/24
export JVM_PATH=/home/s2210405/jdk-19.0.2/lib/server/libjvm.so

CONFIG=( monot5-large-10k_ns_2024 )

N_TRIALS=( 10 )
TMP_DIR=./tmp
VAL_SEGMENT=test

export PYTHONPATH=$PWD

# mkdir -p ./train_logs/tuned

SCRIPT=hp_search.py

for i in ${!CONFIG[@]}
do
    CONFIG_PATH=./configs/tuning/${CONFIG[$i]}.json
    python -u $SCRIPT $CONFIG_PATH ${N_TRIALS[$i]} $TMP_DIR/${CONFIG[$i]} \
        $VAL_SEGMENT $ROOT_DIR | tee ./train_logs/tuned/${CONFIG[$i]}.log
done
