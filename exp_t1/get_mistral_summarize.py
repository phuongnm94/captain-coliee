import os
import re
import gc
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.utils import release_memory

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

os.chdir('/home/s2310409/workspace/coliee-2024/')
from utils.misc import get_summary, get_query

def summarize_prompt(doc):
    prompt = f"""[INST] You are a helpful legal assistant. You are helping a user to summarize case law documents.
    ## Article : \n{doc}"""
    # tokenizer.encode(prompt)
    prompt = prompt + f"\n## TLDR:[\INST]"
    return prompt

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model.eval()

list_files = os.listdir('dataset/processed')
list_files = [f for f in list_files if f.endswith('.txt')]
for file in tqdm(list_files):
    if os.path.exists(f'dataset/mixtral_summarized/{file}'):
        continue
    with open(f'dataset/processed/{file}', 'r') as fp:
        doc = fp.read()
    doc = tokenizer.decode(tokenizer.encode(doc, max_length=9000, truncation=True))
    prompt = summarize_prompt(doc)
    with torch.no_grad():
        encodeds = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    summarized_doc = decoded[0].split('[\INST]')[1].strip()
    with open(f'dataset/mixtral_summarized/{file}', 'w') as fp:
        fp.write(summarized_doc)
