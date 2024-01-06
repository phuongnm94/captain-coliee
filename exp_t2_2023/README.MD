### Instalation
- Run `pip install -r requirements.txt`
### Instructions
- Run `python src/preprocess.py` to create BM25 indexes and generate negative samples
- Run `python train.py --config ./configs/monot5-large-10k_hns.json` (can customize your own config)
