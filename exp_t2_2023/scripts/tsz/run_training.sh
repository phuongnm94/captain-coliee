CONFIG_FILE=$1
SAVE_DIR=$2

export ROOT_DIR=/home/thanhtc/mnt
export NLTK_DATA=$ROOT_DIR/nltk_data/
export JVM_PATH=$ROOT_DIR/packages/jdk-19.0.2/lib/server/libjvm.so

/home/thanhtc/miniconda3/bin/conda activate dev_tsz
deepspeed --num_gpus=1 train.py $CONFIG_FILE -s $SAVE_DIR | tee -a $SAVE_DIR/log
