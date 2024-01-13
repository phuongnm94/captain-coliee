import numpy as np
import pandas as pd
import json
import argparse
import pickle
import torch
import nltk
import os
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

os.chdir('/home/s2310409/workspace/coliee-2024/')

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def load_data(dir):
    with open(dir, 'r') as fp:
        train_data = json.load(fp)

    data = []
    for key in train_data.keys():
        data.append([key, train_data[key]])

    return pd.DataFrame(data, columns=['source', 'target'])

def chunking(sentences, window_size=10):
    chunks = []
    for i in range(0, len(sentences) - window_size, window_size//2):
        chunks.append("\n".join(sentences[i:i+window_size]))
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_candidates', type=int, default=50)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    n_candidates = args.n_candidates
    split = args.split

    word_tokenizer = nltk.tokenize.WordPunctTokenizer()

    with open(f'dataset/{split}.json', 'r') as fp:
        data_dict = json.load(fp)
    
    all_files = []
    for key in data_dict.keys():
        all_files.append(key)
        all_files.extend(data_dict[key])
    
    file_list = list(set(all_files))
    if split == 'test':
        file_list = [f for f in os.listdir(f'dataset/c2023/{split}_files') if f.endswith('.txt')]
    file_list = sorted(file_list)

    processed_file_dict = {}
    for file in [f for f in os.listdir('dataset/processed') if not f.startswith('.')]:
        processed_file = f"dataset/processed/{file}"
        with open(processed_file, 'r') as fp:
            processed_document = fp.read()
            processed_file_dict[file] = {
                'sentences': processed_document.split('\n\n'),
                'processed_document': processed_document
            }

    mode = 'document'
    if mode == 'chunk':
        chunk_dict = {}
        for file in file_list:
            chunks = chunking(processed_file_dict[file]['sentences'])
            for i, chunk in enumerate(chunks):
                if len(chunk) > 0:
                    chunk_dict[f"{file}_{i}"] = chunk
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


    stopwords = nltk.corpus.stopwords.words('english')

    docs = []
    for file in processed_file_dict.keys():
        docs.append(processed_file_dict[file]['processed_document'])

    count_vec = CountVectorizer(stop_words=stopwords)
    word_count_vector = count_vec.fit_transform(docs)


    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    features = count_vec.get_feature_names_out()

    n_keywords = 25

    def extract_query(doc):
        tf_idf_vector=tfidf_transformer.transform(count_vec.transform([doc]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords=extract_topn_from_vector(features,sorted_items, n_keywords)
        return " ".join(list(keywords.keys()))

    query_dict = {}

    for file in tqdm(file_list):
        query_dict[file] = extract_query(processed_file_dict[file]['processed_document'])

    
    candidate_dicts = {}

    for file in tqdm(file_list):
        query = query_dict[file]
        tokenized_query = word_tokenizer.tokenize(query)
        results = bm25.get_scores(tokenized_query)
        max_ids = np.argsort(results)[-n_candidates:]
        document_candidates = [file_list[idx] for idx in max_ids]
        candidate_dicts[file] = list(set(document_candidates))

    with open(f'dataset/c2023/bm25_candidates_{split}_{n_candidates}.json', 'w') as fp:
        json.dump(candidate_dicts, fp)

    test_df = load_data(f'dataset/{split}.json')

    test_df['candidates'] = test_df['source'].apply(lambda x: candidate_dicts[x])
    test_df['query'] = test_df['source'].apply(lambda x: query_dict[x])

    # calculate accuracy metrics for BM25 + TF-IDF
    correct = 0
    n_retrived = 0
    n_relevant = 0

    coverages = []

    for index, row in test_df.iterrows():
        source = row['source']
        target = row['target']
        preds = row['candidates']
        coverages.append(len(preds))
        n_retrived += len(preds)
        n_relevant += len(target)
        for prediction in preds:
            if prediction in target:
                correct += 1

    precision = correct / n_retrived
    recall = correct / n_relevant

    print(f"Average # candidates: {np.mean(coverages)}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {2 * precision * recall / (precision + recall)}")