#!/bin/bash
ROOT_DIR=/home/s2210421

CONFIG=( monot5-large-10k_ns_md monot5-large-10k_ns_iiw_md )

N_TRIALS=( 5 10 )
TMP_DIR=./tmp
VAL_SEGMENT=val

export PYTHONPATH=$PWD

mkdir -p ./train_logs/tuned

SCRIPT=hp_search.py

for i in ${!CONFIG[@]}
do
    CONFIG_PATH=./configs/tuning/${CONFIG[$i]}.json
    /home/s2210421/miniconda3/envs/dev/bin/python -u $SCRIPT $CONFIG_PATH ${N_TRIALS[$i]} $TMP_DIR/${CONFIG[$i]} \
        $VAL_SEGMENT $ROOT_DIR | tee ./train_logs/tuned/${CONFIG[$i]}.log
done
