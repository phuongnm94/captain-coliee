from tqdm import tqdm
import argparse


import xml.etree.ElementTree as Et
import glob
from bs4 import BeautifulSoup
import re
import json
from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

import utils.utils as ult




 
def few_shot_prompting(indexes, corpus, content, labels, prompt_template):
    result = ""
    for i in indexes:
        answer = "Yes"
        if "N" == labels[i]:
            answer = "No"
        prompt = prompt_template.replace("{{premise}}", corpus[i]).replace('{{hypothesis}}', content[i]).replace('{{answer}}', answer)
        result += prompt
    return result
    


def prompting(premise, hypothesis, template=None):
    text = template.replace("{{premise}}", premise).replace("{{hypothesis}}", hypothesis)
    return text

def writefile(data, filename):
    # Serializing json
    json_object = json.dumps(data, indent=1)
    # Writing to sample.json
    with open(filename, "w") as outfile:
        outfile.write(json_object)
        
def few_shot_prompting(indexes, corpus, content, labels, prompt_template):
    result = ""
    for i in indexes:
        if "true or false" in prompt_template.lower():
            answer = "True"
            if "N" == labels[i]:
                answer = "False"
        else:
            answer = "Yes"
            if "N" == labels[i]:
                answer = "No"
        prompt = prompt_template.replace("{{premise}}", corpus[i]).replace('{{hypothesis}}', content[i]).replace('{{answer}}', answer)
        result += prompt
    return result

def predict(model, tokenizer, path_file, passage_embeddings, files=["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], output="../output/accuracy2/newpromt_"):
    for file in files:
        test_file = path_file+file+".xml"
        f = open(path_file+file+".txt", "w", encoding="utf-8")
        data = ult.load_samples(test_file)
        
        acc = {}
        for template_prompt in list_prompt:
            template_prompt = template_prompt["prompt"]
            query_prompt = template_prompt+"\nAnswer: "
            prompt_template = template_prompt+"\nAnswer: {{answer}}\n\n"
            
            result = []
            count = 0
            for item in tqdm(data):
                label = item["label"]
                hypothesis = item["content"]
                premise = item["result"]
                #Important: You must use dot-product, not cosine_similarity
                query_embedding = query_encoder.encode(premise)
                scores = util.dot_score(query_embedding, passage_embeddings)
                indexes = torch.topk(scores, 3).indices[0]
                few_shot = few_shot_prompting(indexes, corpus, content, labels, prompt_template)
                text = few_shot + prompting(premise, hypothesis, query_prompt)
                #############################################
                inputs = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
                outputs = model.generate(inputs, max_new_tokens=10)
                output_text = ult.format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
                f.write(text)
                f.write("\n3-shot answer:"+output_text+"\n")
                f.write("=========================================\n")
                if count < 1:
                    print(text)
                # if count < 3:
                #     print(output_text)
                if "yes" in output_text or "true" in output_text:
                    output_text = "Y"
                else:
                    output_text = "N"
                if output_text == label:
                    count+=1
                # print("predict label: ", output_text, "label: ", label)
            # print("=======================================")
            # print(template_prompt)
            # print(count, "/", len(data))
            acc.update({template_prompt: count/len(data)})
            print(acc)
        writefile(acc, output+file+".json")
        f.close()

import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, required=False)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--list-prompt-path', type=str, required=True)
    parser.add_argument('--test-file-path', type=str, required=True)
    args = parser.parse_args()
    
    test_file = "../../../COLIEE2024statute_data-English/train/riteval_R04_en.xml"
    dev_file = "../../../COLIEE2024statute_data-English/train/riteval_R03_en.xml"


    datas = ult.get_all_files_from_path("../../../COLIEE2024statute_data-English/train")

    corpus = []
    content = []
    labels = []

    for data in datas:
        if "RO4" in data or "R03" in data:
            continue
        print(data)
        data = ult.load_samples(data)
        for item in data:
            corpus.append(item["result"])
            content.append(item["content"])
            labels.append(item["label"])
    print(len(corpus))
    passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')

    passage_embeddings = passage_encoder.encode(corpus)

    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')

    model_name = "google/flan-t5-xxl"
    cache_dir = "/home/s2320037/.cache"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16, load_in_8bit=True
        )

    prompt_template = "{{premise}}\nBased on the previous passage, {{hypothesis}}?\nAnswer: {{answer}}\n\n"
    final_prompt = "{{premise}}\nBased on the previous passage, {{hypothesis}}?\nAnswer: "

    data = ult.load_samples(test_file)

    result = []
    count = 0

    list_prompt = ult.readfile("../data/prompt.json")

    path_file = "../../../COLIEE2024statute_data-English/train/"
    predict(model, tokenizer, path_file, passage_embeddings, ["riteval_R04_en"], "../output/newpromt_")