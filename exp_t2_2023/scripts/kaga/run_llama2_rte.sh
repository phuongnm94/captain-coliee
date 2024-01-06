#!/bin/bash

ROOT_DIR=/home/s2210421
export PYTHONPATH=$ROOT_DIR/projects/llama

DATASET_PATH=$ROOT_DIR/datasets/COLIEE2023/Task2/data_org

IND=1
LLAMA2_MODELS=(../llama/llama-2-7b-chat ../llama/llama-2-13b-chat ../llama/llama-2-70b-chat)
NPROC=(1 2 8)

$ROOT_DIR/miniconda3/envs/dev/bin/python -m torch.distributed.launch \
    --nproc_per_node ${NPROC[$IND]} run_llm.py \
    --task rte \
    --dataset_path $DATASET_PATH \
    --model_path ${LLAMA2_MODELS[$IND]} \
    --tokenizer_path ../llama/tokenizer.model \
    --max_seq_len 2048 \
    --max_output_len 32 \
    --max_batch_size 8 \
    --bm25_index_path ./data/bm25_indexes/coliee_task2/data_org/test \
    --bm25_topk 10 \
    --prompt_id "'9'"