#!/bin/bash

ROOT_DIR=/home/thanhtc/mnt
export NLTK_DATA=/home/thanhtc/mnt/nltk_data/
export JVM_PATH=/home/thanhtc/mnt/packages/jdk-19.0.2/lib/server/libjvm.so
# export TRANSFORMERS_CACHE=/data/huggingface_models
# export HF_HOME=$TRANSFORMERS_CACHE
export PYTHONPATH=$PWD:$ROOT_DIR/projects/llama

DATASET_PATH=$ROOT_DIR/datasets/COLIEE2023/Task2/data_org

IND=1
LLAMA2_MODELS=(../llama/llama-2-7b-chat ../llama/llama-2-13b-chat)
NPROC=(1 2)

torchrun --nproc_per_node ${NPROC[$IND]} src/augment_dataset.py \
    --task summarization \
    --dataset_path $DATASET_PATH \
    --model_path ${LLAMA2_MODELS[$IND]} \
    --tokenizer_path ../llama/tokenizer.model \
    --max_seq_len 2048 \
    --max_batch_size 8 \