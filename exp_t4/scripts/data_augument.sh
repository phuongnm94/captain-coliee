xml_file_paths="/home/s2320037/Collie/COLIEE2024statute_data-English/fewshot" # the path of folder that contains the xml files having queries used for summarization. This folder should contain some R-0x xml files
xml_input_path="/home/s2320037/Collie/COLIEE2024statute_data-English/train/" # the path of folder that contains all the xml files having queries used for training. This folder should contain some R-0x xml files
summarize_model_name="google/flan-t5-xxl"  # name of model used for summarization
summary_output_path="/home/s2320037/Collie/captain-coliee/exp_t4/output/sum/prompt1/" # path of temporary folder for summarization
output_file_name="/home/s2320037/Collie/COLIEE2024statute_data-English/aug_sum/full_data.jsonl" # path to the filename of the final json file
top_k=500
python3 /home/s2320037/Collie/captain-coliee/exp_t4/src/data_argument.py --xml_file_paths $xml_file_paths --summarize_model_name $summarize_model_name \
    --summary_output_path $summary_output_path --output_file_name $output_file_name --top_k $top_k --xml_input_path $xml_input_path