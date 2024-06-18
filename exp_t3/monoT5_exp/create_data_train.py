import re
import xml.etree.ElementTree as Et
import json
import glob
import os
import pandas as pd

from util import load_samples, check_is_usecase
from rank_bm25 import BM25Okapi

full_en_civil_code_df = pd.read_csv("./full_en_civil_code_df.csv")

# full_en_civil_code_df.csv
# article	content
# 0	1	Part I General Provisions Chapter I Common Pro...
# 1	2	Part I General Provisions Chapter I Common Pro...

tokenized_corpus = [doc.split(" ") for doc in full_en_civil_code_df.content.tolist()]
bm25 = BM25Okapi(tokenized_corpus)

to_gen = []
xml_dir = "./COLIEE2023statute_data-English/train"
for file_path in glob.glob(f'{xml_dir}/*.xml'):
    to_gen = to_gen + load_samples(file_path)

gen_df = pd.DataFrame([], columns =['q_id', 'question', 'a_id', 'context', 'label', 'bm25_score'])
for i in range(len(to_gen)):
    index = to_gen[i]['index']
    query = to_gen[i]['content']
    result_ls = to_gen[i]['result']
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    candidate_df = full_en_civil_code_df.copy()
    candidate_df['bm25_score'] = doc_scores
    candidate_df = candidate_df.sort_values(by='bm25_score', ascending=False)

    pos_candidate_df = candidate_df[candidate_df['article'].isin(result_ls)].sort_values(by='bm25_score', ascending=False).copy().reset_index(drop=True)
    pos_row_ls = zip([index]*len(pos_candidate_df), [query]*len(pos_candidate_df), pos_candidate_df.article.tolist(), pos_candidate_df.content.tolist(), [1]*len(pos_candidate_df), pos_candidate_df.bm25_score.tolist())
    pos_gen_df = pd.DataFrame(pos_row_ls, columns =['q_id', 'question', 'a_id', 'context', 'label', 'bm25_score'])

    neg_candidate_df = candidate_df[~candidate_df['article'].isin(result_ls)].sort_values(by='bm25_score', ascending=False).copy().reset_index(drop=True).head(200)
    neg_row_ls = zip([index]*len(neg_candidate_df), [query]*len(neg_candidate_df), neg_candidate_df.article.tolist(), neg_candidate_df.content.tolist(), [0]*len(neg_candidate_df), neg_candidate_df.bm25_score.tolist())
    neg_gen_df = pd.DataFrame(neg_row_ls, columns =['q_id', 'question', 'a_id', 'context', 'label', 'bm25_score'])

    batch_gen_df = pd.concat([pos_gen_df, neg_gen_df]).reset_index(drop=True)
    gen_df = pd.concat([gen_df, batch_gen_df]).reset_index(drop=True)

# q_id	question	a_id	context	label	bm25_score
# 0	H26-1-E	The family court may order the commencement of...	11	Part I General Provisions Chapter II Persons S...	1	30.314748
# 1	H26-1-E	The family court may order the commencement of...	19	Part I General Provisions Chapter II Persons S...	0	38.999214

gen_df.sample(frac=1).reset_index(drop=True).to_csv('gen_df.csv', index=False)