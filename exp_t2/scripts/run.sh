CONFIG_FILE='monot5-large-10k_ns_2024.json'
MODEL_CLASS="monot5"

# Create indexs and generate negative samples with BM25
# python src/preprocess.py

# Fine-tune MonoT5
# python train.py --config_path ./configs/${CONFIG_FILE}

# # Evaluate
python eval_monot5.py \
    --model_dir ./train_logs/${CONFIG_FILE%.*}/ckpt \
    --model ${MODEL_CLASS} \
    --dataset_path ../data/task2_files_2024 \
    --bm25_index_path ../data/bm25_indexes/coliee_task2/2024/val \
    --segment val