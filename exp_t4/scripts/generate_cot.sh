model_name="google/flan-t5-xxl"
cache_dir="/home/s2320037/.cache"
data_path="/home/s2320037/Collie/COLIEE2024statute_data-English/fewshot"
output_cot_path="/home/s2320037/Collie/COLIEE2024statute_data-English/cot"
list_prompt_path="../data/prompt.json"
test_file_path=""
python3 ../src/generated_cot.py --model-name $model_name --cache-dir $cache_dir --data-path $data_path  \
      --list-prompt-path $list_prompt_path --output-cot-path $output_cot_path