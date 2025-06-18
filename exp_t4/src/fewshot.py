
## USE this
from os import listdir
from os.path import isfile, join
import os


import xml.etree.ElementTree as Et
import glob
from bs4 import BeautifulSoup
import re
import json
import utils.utils as ult

from tqdm import tqdm
from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# datas = get_all_files_from_path("../data/COLIEE2024statute_data-English/fewshot")
from sentence_transformers import SentenceTransformer, util

def dpr(testfile, path="../data/COLIEE2024statute_data-English/fewshot"):
    datas = ult.get_all_files_from_path(path)
    corpus = []
    content = []
    labels = []
    for data in datas:
        if testfile in data:
            continue
        data = ult.load_samples(data)
        for item in data:
            # corpus.append(item["result"].replace("\n", " ").strip())
            corpus.append(item["result"].strip())
            content.append(item["content"].strip().replace(".", ""))
            labels.append(item["label"].strip())
    print(len(corpus))
    retrival_passage_embeddings = passage_encoder.encode(corpus)
    content_passage_embeddings = passage_encoder.encode(content)
    return corpus, content, labels, retrival_passage_embeddings, content_passage_embeddings

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

def predict(model, tokenizer, path_file,fewshot_path, files=["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], output="../output/accuracy2/newpromt_"):
    for file in files:
        # test_file = path_file+file+".xml"
        test_file = os.path.join(path_file,file+".xml")
        data = ult.load_samples(test_file)
        corpus, content, labels, retrival_passage_embeddings, content_passage_embeddings = dpr(file,fewshot_path)
        acc = {}
        for template_prompt in list_prompt:
            idx = template_prompt["id"]
            out_path = output+f"prompt_{idx}/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            f_result = open(out_path+file+"_acc.txt", "w", encoding="utf-8")
            f_prompt = open(out_path+file+"_prompt.txt", "w", encoding="utf-8")
            template_prompt = template_prompt["prompt"]
            query_prompt = template_prompt+"\nAnswer: "
            prompt_template = template_prompt+"\nAnswer: {{answer}}\n\n"
            
            result = []
            count = 0
            for item in tqdm(data):
                label = item["label"]
                hypothesis = item["content"]
                premise = item["result"]
                id = item["index"]
                few_shot = ""
                #Important: You must use dot-product, not cosine_similarity
                query_embedding = query_encoder.encode(premise)
                scores = util.dot_score(query_embedding, retrival_passage_embeddings)
                indexes = torch.topk(scores, 3).indices[0]
                few_shot = few_shot_prompting(indexes, corpus, content, labels, prompt_template)
                text = few_shot + prompting(premise, hypothesis, query_prompt)
                #############################################
                inputs = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
                outputs = model.generate(inputs, max_new_tokens=10)
                output_text = ult.format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
                if count < 1:
                    print(text)
                    print(output_text)
                if "yes" in output_text or "true" in output_text:
                    output_text = "Y"
                else:
                    output_text = "N"
                f_result.write(id+"\t"+output_text+"\t"+label+"\n")
                if output_text == label:
                    f_prompt.write(id+": "+text+output_text+"\t"+label+"\n++++++++++++++++++++++++++++++\n")
                    count+=1
                else:
                    f_prompt.write(id+": "+text+output_text+"\t"+label+"\n------------------------------\n")
            acc.update({template_prompt: count/len(data)})
        writefile(acc, out_path+file+".json")


import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, required=False)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--fewshot-path', type=str, required=True)
    parser.add_argument('--list-prompt-path', type=str, required=False)
    parser.add_argument('--test-file-path', type=str, required=False)
    parser.add_argument('--output-file-path', type=str, required=False)
    args = parser.parse_args()

    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
    passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
    # model_name = "google/flan-t5-xxl"
    model_name = args.model_name
    # cache_dir = "/home/congnguyen/drive/.cache"
    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
		model_name, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16, load_in_8bit=True
	)
    # path_file = "../data/COLIEE2024statute_data-English/fewshot/"
    path_file = args.data_path
    # list_prompt = ult.readfile("../data/prompt4.json")
    list_prompt = ult.readfile(args.list_prompt_path)
    # 
    if args.test_file_path is not None:
        testfiles= [args.test_file_path]
    else:
        testfiles = ["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"]
# testfiles = ["riteval_R04_en"]    
    output_file_path = args.output_file_path
    # predict(model, tokenizer, path_file, testfiles, "../output/fewshot_detail/")
    predict(model, tokenizer, path_file,args.fewshot_path, testfiles, output_file_path)