### Instalation
- Run `pip install -r requirements.txt`

### Instructions
- Run `python src/preprocess.py`: create BM25 indexes and generate negative samples
- Run `bash scripts/kaga/run_hp_search.sh`: fine-tuning monoT5 with grid search
- Run `python src/calculate_retrieving_scores.py`: calcualte the BM25 scores and MonoT5 scores for reranking
- Run `python src/fewshot_reranking.py`: rerank with few-shot setting

### Result
| Run        | Dev       | Official Test 2024 |
|------------|-----------|--------------------|
| CAPTAIN    |   75.45   |          -         |
| AMHR       |     -     |      **65.12**     |
| JNLP       |     -     |        63.20       |
| NOWJ       |     -     |        61.17       |
| captainZs2 | **76.36** |        63.35       |
| captainZs3 |   74.55   |        62.35       |
| captainFs2 |   70.13   |        63.60       |