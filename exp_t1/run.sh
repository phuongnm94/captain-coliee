#! /usr/bin/env bash

python process.py
python generate_query.py --n_keywords 25
python split_data.py
python test_bm25.py
