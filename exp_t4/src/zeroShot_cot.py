from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import json
import xml.etree.ElementTree as Et
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import utils.utils as ult


def prompting(premise, hypothesis, template=None):
    text = template.replace("{{premise}}", premise).replace("{{hypothesis}}", hypothesis)
    # return text+"\n\nLet's think step by step\n"
    return text



def predict(model, tokenizer, files=["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], output="../output/cot/newpromt_"):
    for file in files:
        test_file = path_file+file+".xml"
        data = ult.load_samples(test_file)
        
        acc = {}
        for template_prompt in list_prompt:
            idx_prompt = template_prompt["id"]
            template_prompt = template_prompt["prompt"]
            print(template_prompt)
            result = []
            count = 0
            for item in tqdm(data):
                label = item["label"]
                hypothesis = item["content"]
                premise = item["result"]
                #Important: You must use dot-product, not cosine_similarity
                text = prompting(premise, hypothesis, template_prompt)
                inputs = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
                outputs = model.generate(inputs, max_new_tokens=1024)
                output_text = ult.format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
                if count < 1:
                    print(text)
                    print(output_text)
                # if count < 3:
                #     print(output_text)
                if "yes" in output_text or "true" in output_text:
                    output_text = "Y"
                else:
                    output_text = "N"
                if output_text == label:
                    count+=1
                # print("predict label: ", output_text, "label: ", label)
            print("=======================================")
            print(template_prompt)
            print(count, "/", len(data))
            acc.update({template_prompt: count/len(data)})
        ult.writefile(acc, output+file+".json")

# if __name__=="__main__":
#     predict(model, tokenizer, ["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], "../output/zeroshot/promt_idx_prompt")

import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--inference-file', type=str, required=True)
    parser.add_argument('--reference-file', type=str, required=True)
    args = parser.parse_args()

    model_name = "google/flan-t5-xxl"
    cache_dir = "/home/congnguyen/drive/.cache"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16, load_in_8bit=True
        )
    list_prompt = ult.readfile("/home/congnguyen/drive/Coliee2024/data/prompt3.json")

    path_file = "/home/congnguyen/drive/Coliee2024/data/COLIEE2024statute_data-English/fewshot/"
    predict(model, tokenizer, ["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], "../output/zeroshot/promt_idx_prompt")