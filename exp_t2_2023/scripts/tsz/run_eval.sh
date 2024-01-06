#!/bin/bash

MODEL=./train_logs/old/monot5-large-10k_ns_md/3/ckpt/checkpoint-131
# MODEL=castorini/monot5-large-msmarco-10k
TOP_K=1
MARGIN=0
ALPHA=0.9

SEGMENT=test

ROOT_DIR=/home/thanhtc/mnt
export NLTK_DATA=/home/thanhtc/mnt/nltk_data/
export JVM_PATH=/home/thanhtc/mnt/packages/jdk-19.0.2/lib/server/libjvm.so

# python eval_monot5.py bm25 \
#     --dataset_path=/home/s2210421/datasets/COLIEE2023/Task2/data_org \
#     --bm25_index_path=./data/bm25_indexes/coliee_task2/data_org/$SEGMENT \
#     --segment=$SEGMENT -s ./artifacts/coliee_task2/bm25/$SEGMENT.json

python eval_monot5.py monot5 -m $MODEL -k=$TOP_K --margin=$MARGIN --alpha=$ALPHA \
    --dataset_path=$ROOT_DIR/datasets/COLIEE2023/Task2/data_sts \
    --bm25_index_path=./data/bm25_indexes/coliee_task2/data_org/$SEGMENT \
    --segment=$SEGMENT

# python eval_monot5.py monot5 -m $MODEL \
#     --dataset_path=$ROOT_DIR/datasets/COLIEE2023/Task2/data_org \
#     --bm25_index_path=./data/bm25_indexes/coliee_task2/data_org/$SEGMENT \
#     --segment=$SEGMENT

# python eval_monot5.py ens_pred -m $MODEL \
#     --dataset_path=/home/s2210421/datasets/COLIEE2023/Task2/data_org \
#     --pred_file=./results/coliee_task2/data_org/llama-2-7b-chat_bm25_split/lrte_7.json \
#     --segment=$SEGMENT
