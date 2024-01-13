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

# os.chdir('/home/s2310409/workspace/coliee-2024/')

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

stopwords = nltk.corpus.stopwords.words('english')

processed_file_dict = {}
for file in [f for f in os.listdir('dataset/processed') if not f.startswith('.')]:
    processed_file = f"dataset/processed/{file}"
    with open(processed_file, 'r') as fp:
        processed_document = fp.read()
        processed_file_dict[file] = {
            'sentences': processed_document.split('\n\n'),
            'processed_document': processed_document
        }

docs = []
for file in processed_file_dict.keys():
    docs.append(processed_file_dict[file]['processed_document'])

count_vec = CountVectorizer(stop_words=stopwords)
word_count_vector = count_vec.fit_transform(docs)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

features = count_vec.get_feature_names_out()

def extract_query(doc, n_keywords):
    tf_idf_vector=tfidf_transformer.transform(count_vec.transform([doc]))
    sorted_items = sort_coo(tf_idf_vector.tocoo())
    keywords=extract_topn_from_vector(features,sorted_items, n_keywords)
    return " ".join(list(keywords.keys()))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_keywords", type=int, default=25)
    args = parser.parse_args()
    
    n_keywords = args.n_keywords
    
    for file in tqdm(list(processed_file_dict.keys())):
        with open(f"dataset/queries/{file}", 'w') as fp:
            fp.write(extract_query(processed_file_dict[file]['processed_document'], n_keywords))
            