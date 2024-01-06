#!/bin/bash

ROOT_DIR=/home/thanhtc/mnt
export NLTK_DATA=/home/thanhtc/mnt/nltk_data/
export JVM_PATH=/home/thanhtc/mnt/packages/jdk-19.0.2/lib/server/libjvm.so
# export TRANSFORMERS_CACHE=/data/huggingface_models
# export HF_HOME=$TRANSFORMERS_CACHE
export PYTHONPATH=$ROOT_DIR/projects/llama

DATASET_PATH=$ROOT_DIR/datasets/COLIEE2023/Task2/data_org

IND=1
LLAMA2_MODELS=(../llama/llama-2-7b-chat ../llama/llama-2-13b-chat)
NPROC=(1 2)

torchrun --nproc_per_node ${NPROC[$IND]} run_llm.py \
    --task rerank \
    --dataset_path $DATASET_PATH \
    --model_path ${LLAMA2_MODELS[$IND]} \
    --tokenizer_path ../llama/tokenizer.model \
    --max_seq_len 4096 \
    --max_output_len 512 \
    --max_batch_size 1 \
    --bm25_index_path ./data/bm25_indexes/coliee_task2/data_org/test \
    --bm25_topk 5 \
    --skip_exists_results False \
    --prompt_id "'1'"