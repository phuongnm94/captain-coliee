model_name="google/flan-t5-xxl"
cache_dir="/home/s2320037/.cache"
data_path="/home/s2320037/Collie/COLIEE2024statute_data-English/fewshot"
output_file_path="../output/fewshot_cot/"
list_prompt_path="../data/prompt.json"
test_file_path=""
cot_path="/home/s2320037/Collie/COLIEE2024statute_data-English/cot"
python3 ../src/fewshot_cot.py --model-name $model_name --cache-dir $cache_dir --data-path $data_path  \
      --list-prompt-path $list_prompt_path --output-file-path $output_file_path --cot-path $cot_path