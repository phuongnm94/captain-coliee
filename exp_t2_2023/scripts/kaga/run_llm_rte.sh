#!/bin/bash

MODEL=gpt-3.5-turbo

ROOT_DIR=/home/s2210421
# export TRANSFORMERS_CACHE=/home/s2210421/.cache
# export HF_HOME=$TRANSFORMERS_CACHE
# export SENTENCE_TRANSFORMERS_HOME=$TRANSFORMERS_CACHE
# export TORCH_HOME=$TRANSFORMERS_CACHE

DATASET_PATH=$ROOT_DIR/datasets/COLIEE2023/Task2/data_org

$ROOT_DIR/miniconda3/envs/dev/bin/python run_llm.py \
    --task extraction \
    --dataset_path $DATASET_PATH \
    --model_path $MODEL \
    --max_seq_len 512 \
    --max_output_len 32 \
    --max_batch_size 2 \
    --bm25_index_path ./data/bm25_indexes/coliee_task2/data_org/test \
    --bm25_topk 10 \
    --skip_exists_results False \
    --prompt_id "'2'" \


