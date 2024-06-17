# Installation
Install the JDK package
```
wget https://download.oracle.com/java/19/latest/jdk-19_linux-x64_bin.deb

sudo apt-get -qqy install ./jdk-19_linux-x64_bin.deb

sudo update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-19/bin/java 1919
```
Create conda environment
```
conda create -n coliee-task2 py=3.9
conda activate coliee-task2
```

Then, install the required packages via the following command:
```
pip install -r requirements.txt
```

# Usage
## Data preprocessing

The original data of task 2 COLIEE need to be preprocessed in order to prepair for fine-tuning. Run the `preprocess.py` file to process the original data.
For example:
```
python preprocess.py --tmp_dir ../tmp --index_dir ../data/bm25_indices/coliee_task2 --dataset_path ../data/coliee_task2 --neg_save_path ../data
```
 
## Fine-tune model

To fine-tune the MonoT5 model, run the `train.py` file with the config file in the `configs` folder. You can create your own config file.
Here is an example
```
python train.py --config_path ./configs/monot5-large-10k_ns_2024.json
```
## Evaluate model

To evaluate the result by F1-score with the fine-tuned model, run the `eval_monot5.py` file:
```
eval_monot5.py --model_dir <directory to the fine-tuned
```
## Hyper parameters tuning

To optimize the model fine-tuning process, please use hyper parameter tuning. The searching process for hyper parameters is defined in the `hp_run.sh` file. The file structure is as follow:

```
#!/bin/bash

ROOT_DIR=<root directory>
export JVM_PATH=<path-to-libjvm.so-file>

CONFIG=<name-of-the-config-file>

N_TRIALS=<number-of-trials>

TMP_DIR=./tmp

VAL_SEGMENT=<validation segment (train/test)>

export PYTHONPATH=$PWD

mkdir -p ./train_logs/tuned

SCRIPT=hp_search.py

for i in ${!CONFIG[@]}
do
    CONFIG_PATH=./configs/tuning/${CONFIG[$i]}.json
    python -u $SCRIPT $CONFIG_PATH ${N_TRIALS[$i]} $TMP_DIR/${CONFIG[$i]} \
        $VAL_SEGMENT $ROOT_DIR | tee ./train_logs/tuned/${CONFIG[$i]}.log
done
```

To run the hyper parameter search, run the following command:
```
bash hp_run.sh
```