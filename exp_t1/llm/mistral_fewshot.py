from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm
import os
import re
import pandas as pd

os.chdir('/home/s2310409/workspace/coliee-2024/')

def get_summary(doc_name):
    with open(f"dataset/summarized/{doc_name}", 'r') as fp:
        summary = fp.read()
    return summary

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

def load_data(dir):
    with open(dir, 'r') as fp:
        train_data = json.load(fp)

    data = []
    for key in train_data.keys():
        data.append([key, train_data[key]])

    return pd.DataFrame(data, columns=['source', 'target'])

with open('dataset/c2023/bm25_candidates_test.json', 'r') as fp:
    candidate_dict = json.load(fp)

train_df = load_data('dataset/train.json')
with open('dataset/c2023/bm25_candidates_train_50.json', 'r') as fp:
    candidate_dict = json.load(fp)

def preprocess_summary(case):
    doc = get_summary(case)
    doc = re.sub(r'\n', '', doc)
    doc = re.sub(r'\[', '', doc)
    doc = re.sub(r'\]', '', doc)
    doc = doc.strip()
    if doc[-1] != '.':
        doc = doc + '.'
    return doc


def few_shot_prompting(base_content, candidate_content):
    base_example = preprocess_summary('028494.txt')
    positive_example = preprocess_summary('077675.txt')
    negative_example = preprocess_summary('015710.txt')
    
    
    prompt = f"""[INST] You are a helpful legal assistant. You are helping a user to check whether the candidate case is relevant to the base case. So for instance the following:

    ## Base case : {base_example}
    ## Candidate case : {negative_example}
    ## The candidate case is relevant to the base case. True or False? Answer: False.

    ## Base case : {base_example}
    ## Candidate case : {positive_example}
    ## The candidate case is relevant to the base case. True or False? Answer: True.

    ## Base case : {base_content}
    ## Candidate case : {candidate_content}
    ## The candidate case is relevant to the base case. True or False? Answer:"""
    prompt = prompt + "[\INST]"
    return prompt

test_df = load_data('dataset/test.json')
with open('dataset/c2023/bm25_candidates_test_50.json', 'r') as fp:
    candidate_dict = json.load(fp)

prediction_dict = {}
for i in tqdm(range(len(test_df))):
    base_case = test_df['source'][i]

    if f"{base_case.split('.')[0]}.json" in os.listdir('llm/fewshot-result'):
        continue
    
    base_content = preprocess_summary(base_case)
    if len(tokenizer.encode(base_content)) > 2000:
        base_content = tokenizer.decode(tokenizer.encode(base_content, max_length=2000, truncation=True))

    prediction_dict[base_case] = {}
    # group of 5 candidates
    for candidate_case in candidate_dict[base_case]:
        candidate_content = preprocess_summary(candidate_case)
        # truncate candidate content to 2000 tokens:
        if len(tokenizer.encode(candidate_content)) > 2000:
            candidate_content = tokenizer.decode(tokenizer.encode(candidate_content, max_length=2000, truncation=True))
        
        prompt = few_shot_prompting(base_content, candidate_content)
        encodeds = tokenizer(prompt, return_tensors="pt")
        model_inputs = encodeds.to(device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(generated_ids)

        prediction = decoded[0].split('[\INST]')[1].strip()
        prediction_dict[base_case][candidate_case] = prediction
    
    with open(f"llm/fewshot-result/{base_case.split('.')[0]}.json", "w") as fp:
        json.dump(prediction_dict[base_case], fp)

    


