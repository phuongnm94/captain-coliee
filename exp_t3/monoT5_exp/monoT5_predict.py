import sys
sys.path.append("../pygaggle/")
import re
import xml.etree.ElementTree as Et
import json
import glob
import os
import pandas as pd
from tqdm import tqdm
import torch
from util import load_samples, check_is_usecase
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from transformers import T5ForConditionalGeneration

model_name = "./monoT5_model/monot5-large-msmarco-10k"
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
reranker = MonoT5(model=model)

full_en_civil_code_df = pd.read_csv("./full_en_civil_code_df.csv")
candidate_k150_df = pd.read_csv("R05_candidate_k150_24.csv") #from bm25
test_samples = load_samples('./COLIEE2023statute_data-English/test24/riteval_R05_en.xml')

with open('./Task3_eval/test_prediction.txt', 'w') as f:
    for i in tqdm(range(len(test_samples))):
        test_id = test_samples[i]['index']
        query = test_samples[i]['content']
        article_result = test_samples[i]['result']
        candidate_df = candidate_k150_df[candidate_k150_df.query_id == test_id].sort_values(by='mono_score', ascending=False).reset_index(drop=True).head(150)
        candidate_df = candidate_df.sort_values(by='bm25_score', ascending=False).reset_index(drop=True).head(150)
        ranking_scores = reranker.rescore(Query(query), [Text(p[1], {'docid': p[0]}, 0) for p in candidate_df[["article_id", "content"]].values])
        ranking_scores_ls = [r.score for r in ranking_scores]
        candidate_df['mono_score'] = ranking_scores_ls
        top1_candidate_df = candidate_df.head(1).copy()
        thres1 = top1_candidate_df.iloc[0].mono_score*1.8
        top2_candidate_df = candidate_df.copy()
        top2_candidate_df = top2_candidate_df[1:].reset_index(drop=True)
        top2_candidate_df = top2_candidate_df.sort_values(by='mono_score', ascending=False).reset_index(drop=True)
        top2_candidate_df = top2_candidate_df[(top2_candidate_df.mono_score>=thres1)].reset_index(drop=True)
        full_candidate_df = pd.concat([top1_candidate_df, top2_candidate_df]).reset_index(drop=True)
        for j in range (len(full_candidate_df)):
            art_id = full_candidate_df.iloc[j]['article_id']
            bm25_score = full_candidate_df.iloc[j]['mono_score']
            f.write(f'{test_id} Q0 {art_id} {j+1} {bm25_score} CAPTAIN\n')
    