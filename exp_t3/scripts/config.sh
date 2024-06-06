 
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
