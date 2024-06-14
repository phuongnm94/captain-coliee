
---

## Data
Please visit [COLIEE 2023](https://sites.ualberta.ca/~rabelo/COLIEE2024/) for whole dataset request.

Structure of data directory:
```
.
└── dataset/
    ├── bm25/
    │   ├── corpus
    │   └── index
    ├── c2023/
    │   ├── test_files/
    │   │   ├── 000371.txt
    │   │   └── ...
    │   ├── train_files/
    │   │   ├── 000028.txt
    │   │   └── ...
    │   ├── test_no_labels.json
    │   ├── train_labels.json
    │   ├── bm25_candidates_train.json
    │   ├── bm25_candidates_dev.json
    │   └── bm25_candidates_test.json
    ├── c2024/
    │   ├── test_files/
    │   │   ├── 000371.txt
    │   │   └── ...
    │   ├── train_files/
    │   │   ├── 000028.txt
    │   │   └── ...
    │   ├── test_no_labels.json
    │   └── train_labels.json
    ├── processed/
    │   ├── 000028.txt
    │   └── ...
    ├── queries/
    │   ├── 000028.txt
    │   └── ...
    └── summarized/
        ├── 000028.txt
        └── ...
```

## Environments
```bash
conda create -n env_coliee_t1 python=3.10.13
conda activate env_coliee_t1
pip install requirements.txt
```

## All runs
Overall the results can be reproduced by running:
1. Run `python process.py` to generate processed documents.
2. Run `python generate_query.py --n_keywords 25` to generate TF-IDF keywords for each document.
3. Run `split_data.py` to split train, dev, test in the file.
4. Run `test_bm25.py` to checkout BM25's results.
5. Using LLMs to get summary of the documents using FLanT5 in `summary.ipynb`. To get the Mistral summary, run `get_mistrial_summarize.py`.
6. Finetune  monot5 by running `train.sh` or `python train_monot5.py --config ./configs/monot5-large-10k_hns.json`.
7. To evaluate results, run the `mono_t5.ipynb` notebook.
8. To generate submission, run the `get_submission` notebook.

## Results
```
- Top k = 150:
    - bm25 recall 0.8231
    - mono-t5: (f1, precision, recall)
        - dev: 0.3364, 0.3226, 0.3514
        - test: 0.2021, 0.1477, 0.3202
- Top k = 50:
    - bm25 recall 0.6967
    - mono-t5: (f1, precision, recall)
        - dev: 0.3534, 0.3578, 0.3492
        - test: 0.2372, 0.1779, 0.3558
```
# LLMs pair-wise
Check out the `infer_llm.ipynb` notebook.
