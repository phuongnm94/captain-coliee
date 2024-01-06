#!/bin/bash

MODEL=bigscience/T0pp #gpt-3.5-turbo

ROOT_DIR=/home/thanhtc/mnt
export NLTK_DATA=/home/thanhtc/mnt/nltk_data/
export JVM_PATH=/home/thanhtc/mnt/packages/jdk-19.0.2/lib/server/libjvm.so
# export TRANSFORMERS_CACHE=/data/huggingface_models
# export HF_HOME=$TRANSFORMERS_CACHE
export PYTHONPATH=$ROOT_DIR/projects/llama

DATASET_PATH=$ROOT_DIR/datasets/COLIEE2023/Task2/data_org

$ROOT_DIR/miniconda3/envs/dev/bin/python run_llm.py \
    --task rte \
    --dataset_path $DATASET_PATH \
    --model_path $MODEL \
    --max_seq_len 512 \
    --max_output_len 32 \
    --max_batch_size 4 \
    --bm25_index_path ./data/bm25_indexes/coliee_task2/data_org/test \
    --bm25_topk 10 \
    --skip_exists_results False \
    --prompt_id "'9'" \
    # --context_window 3 \
    
