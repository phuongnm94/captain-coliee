#!/usr/bin/bash
# Job Script pbs system 
# Run by: qsub <this_file>

#PBS -N coliee3v2_relevant
#PBS -j oe  
#PBS -q GPU-1
#PBS -o pbs_train_coliee.log
#PBS -e pbs_train_coliee.err.log
#PBS -M phuongnm@jaist.ac.jp 

SCRIPT_DIR=$(dirname "$(realpath $0)")
source ${SCRIPT_DIR}/config.sh
echo $SCRIPT_DIR 
echo $ROOT_DIR

# # +++++++++++ (step 0) +++++++++++
# # setup env 
conda create -n env_coliee python=3.8 --prefix ${ROOT_DIR}/env_coliee/
conda activate ${ROOT_DIR}/env_coliee/
pip install -r requirements.txt

# +++++++++++ (step 1) +++++++++++
# # dataset generation (need to generate 2 times for multiple test/dev files)
python ${ROOT_DIR}/baseline_src/src/data_generator.py \
--path_folder_base ${ROOT_DIR}/data/COLIEE2023statute_data-Japanese/ \
--meta_data_alignment ${ROOT_DIR}/data/COLIEE2023statute_data-English/ \
--path_output_dir ${DATA_DIR} \
--lang jp --topk 150 --type_data task3 --dev_ids R02 --test_ids R03 

python ${ROOT_DIR}/baseline_src/src/data_generator.py \
--path_folder_base ${ROOT_DIR}/data/COLIEE2023statute_data-Japanese/ \
--meta_data_alignment ${ROOT_DIR}/data/COLIEE2023statute_data-English/ \
--path_output_dir ${ROOT_DIR}/data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02r03_r04/ \
--lang jp --topk 150 --type_data task3 --dev_ids R02 R03 --test_ids R04

cp  ${ROOT_DIR}/data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02r03_r04/test.csv ${DATA_DIR}/test_submit.csv
cp  ${ROOT_DIR}/data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02r03_r04/train.csv ${DATA_DIR}/train.csv


# +++++++++++ (step 2) +++++++++++
# enssemble  2 models folder save
ENSS_MODELS="${ROOT_DIR}/settings/Mckpt/models"
mkdir -p $ENSS_MODELS

# train a japanese pretrained lm models 
source ${SCRIPT_DIR}/train.sh cl-tohoku/bert-base-japanese-v2 $ROOT_DIR $DATA_DIR
cd $ENSS_MODELS 
for ckpt in ${MODEL_OUT}/*ckpt  ${MODEL_OUT}/*pred.pkl; do  
    ln -s $ckpt 
done

# # train a japanese pretrained lm models 
source ${SCRIPT_DIR}/train.sh cl-tohoku/bert-base-japanese-whole-word-masking $ROOT_DIR $DATA_DIR
cd $ENSS_MODELS 
for ckpt in ${MODEL_OUT}/*ckpt ${MODEL_OUT}/*pred.pkl; do
    ln -s $ckpt  
done 

# +++++++++++ (step 3) +++++++++++
# evaluate setting Mckpt
python ${ROOT_DIR}/baseline_src/src/train.py \
--data_dir ${DATA_DIR}/ --model_name_or_path cl-tohoku/bert-base-japanese-v2 --no_train \
--log_dir ${ENSS_MODELS}/ --file_output_id bjpAll \
 > ${ENSS_MODELS}/eval.log && cat ${ENSS_MODELS}/eval.log


# +++++++++++ (step 4) +++++++++++
# LLM inference with prompting technique
python ${ROOT_DIR}/baseline_src/src/llm_infer.py \
 --data_dir ${ROOT_DIR}/data/ \
 --plm_model_prediction_path ${ROOT_DIR}/settings/Mckpt/models \
 --llm_model_name google/flan-t5-xxl 