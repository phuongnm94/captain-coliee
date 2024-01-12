import os
import re
import copy
import random
import collections
import torch
import numpy as np
import pandas as pd
import json
import pickle
import nltk

from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (TrainerCallback, AutoTokenizer, Trainer, Seq2SeqTrainer, 
                          TrainingArguments, AutoModelForSeq2SeqLM)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from utils.misc import load_data, get_query, get_summary

# os.chdir('/home/s2310409/workspace/coliee-2024/')

def chunking(sentences, window_size=10):
    chunks = []
    for i in range(0, len(sentences) - window_size, window_size//2):
        chunks.append("\n".join(sentences[i:i+window_size]))
    return chunks

# BUILD DATASET

def split_target(df):
        data = []
        for i in range(len(df)):
            for j in range(0, len(df['target'][i]), 3):
                data.append([df['source'][i], df['target'][i][j:j+3]])
        return pd.DataFrame(data, columns=['source', 'target'])

def build_dataset(use_chunk=False, mode='train', n_candidates=50):

    word_tokenizer = nltk.tokenize.WordPunctTokenizer()
    # file_list = sorted(list(all_data_dict.keys()))

    file_list = [f for f in os.listdir(f'dataset/c2023/{mode}_files') if f.endswith('.txt')]
    file_list = sorted(file_list)

    processed_file_dict = {}
    for file in [f for f in os.listdir("dataset/processed") if not f.startswith('.')]:
        processed_file = f"dataset/processed/{file}"
        with open(processed_file, 'r') as fp:
            processed_document = fp.read()
            processed_file_dict[file] = {
                'sentences': processed_document.split('\n\n'),
                'processed_document': processed_document
            }

    chunk_dict = {}
    for file in file_list:
        chunks = chunking(processed_file_dict[file]['sentences'])
        for i, chunk in enumerate(chunks):
            if len(chunk) > 0:
                chunk_dict[f"{file}_{i}"] = chunk

    if use_chunk:
        # bm25 for chunks
        corpus = []
        chunk_list = sorted(list(chunk_dict.keys()))
        for chunk in chunk_list:
            corpus.append(chunk_dict[chunk])
        tokenized_corpus = [word_tokenizer.tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
    else:
        # bm25 for whole document
        corpus = []
        prcessed_list = sorted(file_list)
        for file in prcessed_list:
            corpus.append(processed_file_dict[file]['processed_document'])
        tokenized_corpus = [word_tokenizer.tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

    candidate_dicts = {}

    for file in tqdm(file_list):
        query = get_query(file)
        tokenized_query = word_tokenizer.tokenize(query)
        results = bm25.get_scores(tokenized_query)
        max_ids = np.argsort(results)[-n_candidates:]
        document_candidates = [file_list[idx] for idx in max_ids]
        candidate_dicts[file] = list(set(document_candidates))

    data_df = load_data(f'dataset/json/{mode}.json')
    data_df = split_target(data_df)

    data_df['candidates'] = data_df['source'].apply(lambda x: candidate_dicts[x])
    data_df['query'] = data_df['source'].apply(lambda x: get_query(x))

    dataset = []
    for index, row in data_df.iterrows():
        query = row['query']
        source = row['source']
        targets = row['target']
        candidates = row['candidates']

        neg_candidates = [c for c in candidates if c not in targets+[source]]
        
        case_dict = {
            'id': source,
            'text': query, 
            'pos_candidates': [{"id": {p_id}, 
                                "text": get_summary(p_id)} for p_id in targets],
            'neg_candidates': [{"id": {p_id}, 
                                "text": get_summary(p_id)} for p_id in neg_candidates]
        }
        dataset.append(case_dict)
    return dataset


class TrainDataset(Dataset):
    def __init__(self, dataset, num_pairs_per_batch, ns_strategy,
                 num_msmarco_pairs_per_batch=None, enhanced_weight=None, train_uncased=False):
        self.data = dataset
        self.training_data = []

        self.ps = {sample["id"]: [pos["id"] for pos in sample["pos_candidates"]] for sample in self.data}
        self.ps_iter = copy.deepcopy(self.ps)

        self.ns = {sample["id"]: [neg["id"] for neg in sample["neg_candidates"]] for sample in self.data}
        self.ns_iter = copy.deepcopy(self.ns)

        self.use_msmarco_dataset = num_msmarco_pairs_per_batch is not None
        # if self.use_msmarco_dataset:
        #     self.ms_training_data = get_msmarco_dataset(root_dir)
        #     self.num_msmarco_pairs_per_batch = num_msmarco_pairs_per_batch
        #     self.num_pairs_per_batch = num_pairs_per_batch - num_msmarco_pairs_per_batch
        # else:
        #     self.num_pairs_per_batch = num_pairs_per_batch
        self.num_pairs_per_batch = num_pairs_per_batch
        self.use_instance_weighting = enhanced_weight is not None
        self.enhanced_weight = enhanced_weight

    def __len__(self):
        return len(self.data)

    def create_training_dataset(self, model, epoch):
        np.random.shuffle(self.data)
        self.hard_negative_sampling_dataset(model)
        
    def __getitem__(self, idx):
        return self.training_data[idx]
    
    @torch.no_grad()
    def hard_negative_sampling_dataset(self, model):
        model.eval()
        reranker = MonoT5(model=model)

        self.training_data = []
        for sample in tqdm(self.data):
            batch = []
            texts = []
            for i, cand in enumerate(sample["neg_candidates"]):
                texts.append(Text(cand["text"], metadata={"id": cand["id"]}))
            if self.use_instance_weighting:
                for i, cand in enumerate(sample["pos_candidates"]):
                    texts.append(Text(cand["text"], metadata={"id": cand["id"]}))

            query = Query(sample["text"])
            results = reranker.rerank(query, texts)
            if self.use_instance_weighting:
                pos_ids = [cand["id"] for cand in sample["pos_candidates"]]
                neg_ids = [cand["id"] for cand in sample["neg_candidates"]]

                top_idxs = [c.metadata["id"] for c in results][:self.num_pairs_per_batch]
                last_pos_idx, top_neg_idx = -1, 100
                for i, idx in enumerate(top_idxs):
                    if idx in pos_ids:
                        last_pos_idx = i
                    if idx in neg_ids and top_neg_idx == 100:
                        top_neg_idx = i  

                for i, cand in enumerate(sample["pos_candidates"]):
                    if cand["id"] not in top_idxs or \
                            (cand["id"] in top_idxs and top_idxs.index(cand["id"]) > top_neg_idx):
                        cand_w = self.enhanced_weight
                    else:
                        cand_w = 1
                    batch.append((sample["text"], cand["text"], 1, cand_w))

                num_neg_pairs = max(self.num_pairs_per_batch - len(sample["pos_candidates"]), 0)
                neg_cands = [cand for cand in sample["neg_candidates"] if cand["id"] in top_idxs]
                for i, cand in enumerate(neg_cands):
                    if top_idxs.index(cand["id"]) < last_pos_idx:
                        cand_w = self.enhanced_weight
                    else:
                        cand_w = 1
                    batch.append((sample["text"], cand["text"], 0, cand_w))
            else:
                for i, cand in enumerate(sample["pos_candidates"]):
                    batch.append((sample["text"], cand["text"], 1, 1))

                num_neg_pairs = max(self.num_pairs_per_batch - len(sample["pos_candidates"]), 0)
                top_neg_idxs = [c.metadata["id"] for c in results][:num_neg_pairs]
                neg_cands = [cand for cand in sample["neg_candidates"] if cand["id"] in top_neg_idxs]
                for i, cand in enumerate(neg_cands):
                    batch.append((sample["text"], cand["text"], 0, 1))

            if self.use_msmarco_dataset:
                ms_data_ind = np.random.choice(len(self.ms_training_data),
                                               self.num_msmarco_pairs_per_batch)
                for i in ms_data_ind:
                    batch.append((*self.ms_training_data[i][0], 1))
                    batch.append((*self.ms_training_data[i][1], 1))
            self.training_data.append(batch)

            model.train()

    @torch.no_grad()
    def random_negative_sampling_dataset(self, model):
        model.eval()

        self.training_data = []
        for sample in self.data:
            batch = []

            for cand in sample["pos_candidates"]:
                batch.append((sample["text"], cand["text"], 1, 1))

            num_neg_pairs = max(self.num_pairs_per_batch - len(sample["pos_candidates"]), 0)
            if len(sample["neg_candidates"]):
                neg_cands_id = np.random.choice(len(sample["neg_candidates"]), num_neg_pairs,
                                                replace=True)
                neg_cands = [sample["neg_candidates"][i] for i in neg_cands_id]
                for cand in neg_cands:
                    batch.append((sample["text"], cand["text"], 0, 1))

            if self.use_msmarco_dataset:
                ms_data_ind = np.random.choice(len(self.ms_training_data),
                                               self.num_msmarco_pairs_per_batch)
                for i in ms_data_ind:
                    batch.append((*self.ms_training_data[i][0], 1))
                    batch.append((*self.ms_training_data[i][1], 1))
            self.training_data.append(batch)

            model.train()

