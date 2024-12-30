ROOT_DIR=./

cd $ROOT_DIR

# python src/preprocess.py

bash scripts/kaga/run_hp_search.sh

# python retrieval_scoring.py
# python fewshot_reranking.py