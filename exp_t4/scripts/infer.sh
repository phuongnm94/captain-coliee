# TESTSET=riteval_R04_en.xml
# TEAM=JNLP3_TestData_en
# INFER=R04.task4.$TEAM.xml


# python src/evaluate.py \
#     --inference-file 2023.Zero_shot_Task4/submit/$INFER \
#     --reference-file data/COLIEE2024statute_data-English/train/$TESTSET

# # python src/evaluate.py \
# #     --inference-file 2023.Zero_shot_Task4/submit/R04.task4.JNLP2_TestData_en.xml \
# #     --reference-file data/COLIEE2024statute_data-English/$TESTSET

# # "OpenPipe/mistral-ft-optimized-1218"
# # "declare-lab/flan-alpaca-xxl"
# MODEL="google/flan-t5-xxl" 
# MODEL="WizardLM/WizardLM-13B-V1.2" 
# MODEL="/home/congnguyen//huggingface/llama-2/llama-2-13b-chat"

CUDA_VISIBLE_DEVICES=0 python src/infer.py --model_name "declare-lab/flan-alpaca-xxl" \
     --prompt_file data/prompt.json --output_folder output/ \
     --cache_dir /home/congnguyen/drive/.cache --file R04

CUDA_VISIBLE_DEVICES=0 python src/infer.py --model_name "declare-lab/flan-alpaca-xxl" \
     --prompt_file data/prompt.json --output_folder output/ \
     --cache_dir /home/congnguyen/drive/.cache --file R03

CUDA_VISIBLE_DEVICES=0 python src/infer.py --model_name "declare-lab/flan-alpaca-xxl" \
     --prompt_file data/prompt.json --output_folder output/ \
     --cache_dir /home/congnguyen/drive/.cache --file R02

CUDA_VISIBLE_DEVICES=0 python src/infer.py --model_name "declare-lab/flan-alpaca-xxl" \
     --prompt_file data/prompt.json --output_folder output/ \
     --cache_dir /home/congnguyen/drive/.cache --file R01

CUDA_VISIBLE_DEVICES=0 python src/infer.py --model_name "google/flan-t5-xxl" \
     --prompt_file data/prompt.json --output_folder output/ \
     --cache_dir /home/congnguyen/drive/.cache --file R04

CUDA_VISIBLE_DEVICES=0 python src/infer.py --model_name "google/flan-t5-xxl" \
     --prompt_file data/prompt.json --output_folder output/ \
     --cache_dir /home/congnguyen/drive/.cache --file R03

CUDA_VISIBLE_DEVICES=0 python src/infer.py --model_name "google/flan-t5-xxl" \
     --prompt_file data/prompt.json --output_folder output/ \
     --cache_dir /home/congnguyen/drive/.cache --file R02

CUDA_VISIBLE_DEVICES=0 python src/infer.py --model_name "google/flan-t5-xxl" \
     --prompt_file data/prompt.json --output_folder output/ \
     --cache_dir /home/congnguyen/drive/.cache --file R01