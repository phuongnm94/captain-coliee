from os import listdir
from os.path import isfile, join



from bs4 import BeautifulSoup
import re
import json

import utils.utils as ult

import xml.etree.ElementTree as Et
import glob

from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

def load_jsonl(file):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    return data
f = open("../data/COLIEE2024statute_data-English/text/civil_code_en-1to724-2.txt", "r")

articles_list = []
articles = {}
pre_art = ""
result = ""
for line in f:
    line = line.strip()
    if line.split()[0] == "Article":
        if result != "":
            articles_list.append(result.strip().replace("  ", " "))
            articles.update({result.strip().split("  ")[0]: result.strip().replace("  ", " ")})
            result = line+" "
        else:
            result += line+" "
    elif line[0] == "(":
        if line[1].isupper():
            continue
        else:
            result += line+" "

f.close()
# articles



model_name = "Qwen/Qwen-72B-Chat"
cache_dir = "/home/congnguyen/drive/.cache"
# cache_dir = ".cache"

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-72B-Chat", cache_dir=cache_dir, trust_remote_code=True)
# device_map="auto", 
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-72B-Chat", cache_dir=cache_dir, device_map="auto", 
                                             torch_dtype=torch.float16, trust_remote_code=True, load_in_4bit=True).eval()
                                

prompt = "'{{text}}'. Analyze the structure following main premise, exception of each rule."
result_art = {}
# history = "Main Clause: A juridical person is not formed other than pursuant to the provisions of this Code or other laws.\nException Clauses:None"
for art in articles:
    result_art.update({art: "Error OOM!"})
    text = prompt.replace("{{text}}", articles[art].strip())
    try:
        response, history = model.chat(tokenizer, text, history=None)
        print(response)
        result_art.update({art: response})
    except:
        ult.write_json("../data/COLIEE2024statute_data-English/text/result_art_fujisan.json", result_art)

import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--inference-file', type=str, required=True)
    parser.add_argument('--reference-file', type=str, required=True)
    args = parser.parse_args()