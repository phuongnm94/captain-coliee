#!/bin/bash

# ROOT_DIR=/home/thanhtc/mnt
# export NLTK_DATA=$ROOT_DIR/nltk_data/
# export JVM_PATH=$ROOT_DIR/packages/jdk-19.0.2/lib/server/libjvm.so
ROOT_DIR=/home/s2420414

# CONFIG=( monot5-large-10k_ns_iiw_md )
CONFIG=(monot5-large-10k_hns)

N_TRIALS=( 10 )
TMP_DIR=./tmp
VAL_SEGMENT=val

export PYTHONPATH=$PWD

mkdir -p ./train_logs/tuned

SCRIPT=hp_search.py

for i in ${!CONFIG[@]}
do
    # CONFIG_PATH=./configs/tuning/${CONFIG[$i]}.json
    CONFIG_PATH=./configs/${CONFIG[$i]}.json
    python -u $SCRIPT $CONFIG_PATH ${N_TRIALS[$i]} $TMP_DIR/${CONFIG[$i]} \
        $VAL_SEGMENT $ROOT_DIR | tee ./train_logs/tuned/${CONFIG[$i]}.log
done
