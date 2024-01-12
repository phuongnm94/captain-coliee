import os
import re
import sys
import json
import string
import pandas as pd

sys.path.append("modules/pygaggle")

# CACHE_DIR = "/home/s2210421/.cache"
# os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
# os.environ["HF_HOME"] = CACHE_DIR
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR
# os.environ['TORCH_HOME'] = CACHE_DIR

import spacy
# from nltk import corpus
# ENG_WORDS = set(corpus.words.words())
# STOPWORDS = set(corpus.stopwords.words('english'))

SPECIAL_CHARACTERS = "/-'#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + \
    '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
PUNCTUATION = ".,!?"

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


def int2str(i):
    return "0" * (3 - len(str(i))) + str(i)


def is_word(word):
    return all(c in string.ascii_lowercase for c in word)


def save_txt(file_path, text):
    with open(file_path,"w", encoding="utf-8") as f:
        f.write(text)


def load_txt(file_path, skip=0):
    with open(file_path, encoding="utf-8") as f:
        while skip > 0:
            f.readline()
            skip -= 1
        data = f.read()
    return data


def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(file_path, d):
    with open(file_path, "w+", encoding="utf-8") as f:
        json.dump(d, f)


def append_json(file_path, d):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d) + "\n")


def get_sentences(doc):
    doc = nlp(doc)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


def filter_document(doc, min_sentence_length=None):
    sentences = get_sentences(doc)
    if min_sentence_length:
        sentences = [sent for sent in sentences
                     if len(sent.split()) >= min_sentence_length]
    doc = " ".join(sentences)
    return doc


def segment_document(doc, max_sent_per_segment, stride, max_segment_len=None):
    sentences = get_sentences(doc)
    segments = []
    for i in range(0, len(sentences), stride):
        segment = " ".join(sentences[i:i + max_sent_per_segment])

        if max_segment_len:
            segment = " ".join(segment.split()[:max_segment_len])
        segments.append(segment)
    return segments

def get_summary(doc_name):
    with open(f"dataset/summarized/{doc_name}", 'r') as fp:
        summary = fp.read()
    return summary

def get_query(doc_name):
    with open(f"dataset/queries/{doc_name}", 'r') as fp:
        query = fp.read()
    return query

def load_data(dir):
    with open(dir, 'r') as fp:
        train_data = json.load(fp)

    data = []
    for key in train_data.keys():
        data.append([key, train_data[key]])

    return pd.DataFrame(data, columns=['source', 'target'])