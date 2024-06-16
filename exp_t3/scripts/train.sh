#!/usr/bin/bash
# Job Script pbs system 
# Run by: qsub <this_file>

#PBS -N coliee3v2_relevant
#PBS -j oe  
#PBS -q GPU-1
#PBS -o pbs_train_coliee.log
#PBS -e pbs_train_coliee.err.log
#PBS -M phuongnm@jaist.ac.jp 

cd $PBS_O_WORKDIR
source ~/.bashrc
SCRIPT_DIR=$(dirname "$(realpath $0)")
echo $SCRIPT_DIR

# this value canbe replaced by a path of downloded model (special for japanese pretrained model)
MODEL_NAME=${1:-"cl-tohoku/bert-base-japanese-whole-word-masking"}  # cl-tohoku/bert-base-japanese-v2  
ROOT_DIR=${2:-"${SCRIPT_DIR}/../"} 
DATA_DIR=${3:-"${ROOT_DIR}/data/COLIEE2023statute_data-Japanese/data_ja_topk_150_r02_r03/"}  

MAX_EP=3
MAX_SEQ=512
LR=5e-6

IFS='/' read -ra MOD_INFO <<< "$MODEL_NAME"
MODEL_ID_1=${MOD_INFO[1]}
SETTING_NAME="${MODEL_ID_1}_top150-newE${MAX_EP}Seq${MAX_SEQ}L${LR}" 
SETTING_DIR="${ROOT_DIR}/settings/${SETTING_NAME}/" 
CODE_DIR="${ROOT_DIR}/baseline_src/src/" 
MODEL_OUT="${SETTING_DIR}/models"
conda activate ${ROOT_DIR}/env_coliee

CMD="CUDA_VISIBLE_DEVICES=0 && cd $CODE_DIR && python ${ROOT_DIR}/baseline_src/src/train.py \
  --data_dir  $DATA_DIR/ \
  --model_name_or_path $MODEL_NAME \
  --log_dir $MODEL_OUT \
  --max_epochs $MAX_EP \
  --batch_size 16 \
  --max_keep_ckpt 5 \
  --lr $LR \
  --gpus 0 \
  --max_seq_length $MAX_SEQ \
  > $MODEL_OUT/train.log "

cd $ROOT_DIR
mkdir ${ROOT_DIR}/settings/ $SETTING_DIR $MODEL_OUT

echo $CMD
eval $CMD
  

  
wait
echo "All done"

