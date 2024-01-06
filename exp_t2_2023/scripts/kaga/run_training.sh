CONFIG_FILE=$1
SAVE_DIR=$2

/home/s2210421/miniconda3/bin/conda activate dev
deepspeed --num_gpus=1 train.py $CONFIG_FILE -s $SAVE_DIR | tee -a $SAVE_DIR/log