tokenize_model="google/flan-t5-xxl"
model_name="philschmid/flan-t5-xxl-sharded-fp16"
cache_dir="/home/s2320037/.cache"
output_path="../output/finetuned"
list_prompt_path="../data/prompt.json"
test_file_path="/home/s2320037/Collie/COLIEE2024statute_data-English/test"
train_file_path="/home/s2320037/Collie/COLIEE2024statute_data-English/train"
output_data_path=$output_path/data
batch_size=32
prompt_id=20
epochs=100
output_name_project="finetuned-flan-t5-xxl"
strategy="zeroshot_aug_r32"
# strategy="fewshot"
fewshot_data_path="/home/s2320037/Collie/COLIEE2024statute_data-English/cot"
python3 ../src/finetune.py --model-name $model_name --cache-dir $cache_dir  --strategy  $strategy \
      --list-prompt-path $list_prompt_path --output-path $output_path --train-file-path $train_file_path --test-file-path $test_file_path \
    --batch_size $batch_size --epochs $epochs --prompt-id $prompt_id --output-name-project $output_name_project