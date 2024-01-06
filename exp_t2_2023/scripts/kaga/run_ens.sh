#!/bin/bash

# MODELS=( "./train_logs/tuned/monot5-base"  "./train_logs/tuned/monot5-base-10k"
#          "./train_logs/tuned/monot5-large" "./train_logs/tuned/monot5-large-10k" )
MODELS=( "" )

for i in ${!MODELS[@]}
do
    if [[ "${MODELS[$i]}" == "" ]]; then
        python -u evaluate.py ens --segment test --optuna=0 -s ./train_logs/ens/monot5_large_no_test_10-ckpts_grid.json \
            | tee ./train_logs/ens/monot5_large_no_test_10-ckpts_grid.log
    else
        python -u evaluate.py ens --segment val -m ${MODELS[$i]} | tee ${MODELS[$i]}/ens_log
    fi
done
