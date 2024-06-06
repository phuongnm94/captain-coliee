
---

## Data
Please visit [COLIEE 2023](https://sites.ualberta.ca/~rabelo/COLIEE2024/) for whole dataset request.

- structure of data directory (same structure between enlgish and japanese datasets)
    ```
        data/COLIEE2024statute_data-English/ 
        └── ... # (similar to Japanese data structure)
        data/COLIEE2024statute_data-Japanese/
        ├── data_jp_topk_150_r02_r03 # folder save output of data_generator
        │   ├── dev.csv
        │   ├── stats.txt
        │   ├── test.csv
        │   ├── tfidf_classifier.pkl
        │   ├── all_data.json
        │   └── train.csv
        ├── text
        │   └── civil_code_jp-1to724-2.txt
        └── train
            ├── riteval_H18_en.xml
            ├── ...
            ├── riteval_R02_jp.xml
            ├── riteval_R03_jp.xml
            └── riteval_R04_jp.xml
    ```

## Environments
```bash 
conda create -n env_coliee python=3.8
conda activate env_coliee
pip install -r requirements.txt
```

## All runs: 
1. Use vscode debugging for better checking runs: config in file `.vscode/launch.json`
2. **Runs**:
   1. Overall, the result can be reproduced by:
      ```cmd
      bash scripts/run.sh
      ```
   2. Explaination each steps:
      1. generate data, extract raw data from COLIEE competition to the `.json` and `.csv` data for training process: 
            
            ```bash
            # +++++++++++ (step 1) +++++++++++
            # dataset generation (need to generate 2 times for multiple test/dev files)
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

            ``` 
            Output (recall score is important)
            ```
            ...
            127 16350 131 P:  0.0077675840978593275 R:  0.9694656488549618 F1:  0.015411686184090771 F2:  0.037631859665757966
            [W] Learning Tfidf Vectorizer ...
            970 120900 1040 P:  0.008023159636062862 R:  0.9326923076923077 F1:  0.0159094636706577 F2:  0.03878138493523109
            Number data pairs:  120970
            127 16350 131 P:  0.0077675840978593275 R:  0.9694656488549618 F1:  0.015411686184090771 F2:  0.037631859665757966
            Number data pairs:  16350
            99 12150 101 P:  0.008148148148148147 R:  0.9801980198019802 F1:  0.01616194596359481 F2:  0.039429663852158674
            Number data pairs:  12150
            len(train_data_pairs_id), len(test_data_pairs_id), len(dev_data_pairs_id) =  120970 16350 12150
            ```
      2. train model by finetuning BERT or pretrained Japanese models (https://huggingface.co/cl-tohoku/bert-base-japanese-v2 or https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking )
            ```bash
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
            ```
            after run all, all aodel checkpoint are generated, the enssemble method is output of setting `Mckpt` in paper. 
      3. infer and evaluation 
            ```bash
            # +++++++++++ (step 3) +++++++++++
            # evaluate setting Mckpt
            python ${ROOT_DIR}/baseline_src/src/train.py \
            --data_dir ${DATA_DIR}/ --model_name_or_path cl-tohoku/bert-base-japanese-v2 --no_train \
            --log_dir ${ENSS_MODELS}/ --file_output_id bjpAll \
            > ${ENSS_MODELS}/eval.log && cat ${ENSS_MODELS}/eval.log
            ```
            The output is saved in ```${ENSS_MODELS}/eval.log```
      4. llm inference
            ```bash 
            # +++++++++++ (step 4) +++++++++++
            # LLM inference with prompting technique
            python ${ROOT_DIR}/baseline_src/src/llm_infer.py \
            --data_dir ${ROOT_DIR}/data/ \
            --plm_model_prediction_path ${ROOT_DIR}/settings/Mckpt/models \
            --llm_model_name google/flan-t5-xxl >${ENSS_MODELS}/eval_llm.log
            ```
            The output is saved in ```${ENSS_MODELS}/eval_llm.log```

