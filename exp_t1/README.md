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
5. Finetune  monot5 by running `train.sh` or `python train_monot5.py --config ./configs/monot5-large-10k_hns.json` 
6. Eval monot5: [PENDING]

# LLMs pair-wise
[PENDING]
