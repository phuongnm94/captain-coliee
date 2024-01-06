#!/bin/bash

MODEL=./train_logs/tuned/monot5-large-10k/ckpt/checkpoint-65
TOP_K=1
MARGIN=0
ALPHA=0.9

# python eval_monot5.py bm25 \
#     --dataset_path=/home/s2210421/datasets/COLIEE2023/Task2/data_org \
#     --bm25_index_path=./data/bm25_indexes/coliee_task2/data_org/test \
#     --segment=test -s ./artifacts/coliee_task2/bm25/test.json

python eval_monot5.py monot5 -m $MODEL -k=$TOP_K --margin=$MARGIN --alpha=$ALPHA \
    --dataset_path=/home/s2210421/datasets/COLIEE2023/Task2/data_extracted_sts \
    --bm25_index_path=./data/bm25_indexes/coliee_task2/data_org/test \
    --segment=test

# python eval_monot5.py monot5 -m $MODEL \
#     --dataset_path=/home/s2210421/datasets/COLIEE2023/Task2/data_org \
#     --bm25_index_path=./data/bm25_indexes/coliee_task2/data_org/test \
#     --segment=test

# python eval_monot5.py ens_pred -m $MODEL \
#     --dataset_path=/home/s2210421/datasets/COLIEE2023/Task2/data_org \
#     --pred_file=./results/coliee_task2/data_org/llama-2-7b-chat_bm25_split/lrte_7.json \
#     --segment=test
