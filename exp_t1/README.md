# Prepare the data directory:
```
dataset
    - bm25 // to store bm25 indices
    - c2023
        - test_files
        - train_files
        - test_no_labels.json
        - train_labels.json
    - c2024
        - test_files
        - train_files
        - test_no_labels.json
        - train_labels.json
    - json
    - processed // store processed documents
    - queries // store extracted TF-IDF keywords using as queries
    - summarized
```

# Mono T5
1. Run `python process.py` to generate processed documents.
2. Run `python generate_query.py --n_keywords 25` to generate TF-IDF keywords for each document.
3. Split train, dev, test in the file `split_data.ipynb.`
4. Check out bm25 result in `test_bm25.ipynb`
5. Using LLMs to get summary of the documents in `summary.ipynb`
6. Finetune  monot5 by running `train.sh` or `python train_monot5.py --config ./configs/monot5-large-10k_hns.json` 
7. Eval monot5: run the `mono_t5.ipynb` notebook

Results:
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

# LLMs pair-wise
Check out the `infer_llm.ipynb` notebook.
