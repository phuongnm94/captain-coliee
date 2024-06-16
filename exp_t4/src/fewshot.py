from os import listdir
from os.path import isfile, join
import os

def get_all_files_from_path(mypath):
    filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return filenames

from bs4 import BeautifulSoup
import re
import json

def get_article(articles):
    result = {}
    current_statue = "(non-statute)"
    for i in re.split(r"(.*)", articles.strip()):
        if len(i) == 0 or i == "\n":
            continue
        if re.search(r"^\(.*\)$", i):
            current_statue = i.strip()
            if current_statue not in result:
                result.update({current_statue: []})
        else:
            if current_statue not in result:
                result.update({current_statue: []})
            result[current_statue].append(i)
    return result

def build_test(filename):
    result = {}
    with open(filename, 'r') as f:
        data = f.read()

    data = BeautifulSoup(data, "xml").find_all('pair')
    for i in data:
        id = i.get('id')
        result.update({id: {}})
        result[id].update({"label": i.get('label')})
        articles = i.find('t1').text.strip()
        # articles = get_article(articles)
        result[id].update({"result": articles})
        result[id].update({"content": i.find('t2').text.strip()})
    return result

def write_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

import xml.etree.ElementTree as Et
import glob

def format_first_line(text):
    lines = text.split("\n")
    results = []
    for line in lines:
        if line[0] == "":
            continue
        if line[0] == "(" and line[-1] == ")":
            continue
        results.append(line)
    return "\n".join(results)

def load_samples(filexml):
    # try:
        tree = Et.parse(filexml)
        root = tree.getroot()
        samples = []
        for i in range(0, len(root)):
            sample = {'result': []}
            for j, e in enumerate(root[i]):
                if e.tag == "t1":
                    sample['result'] = format_first_line(e.text.strip())
                elif e.tag == "t2":
                    question = e.text.strip()
                    sample['content'] = question if len(question) > 0 else None
            sample.update(
                {'index': root[i].attrib['id'], 'label': root[i].attrib.get('label', "N")})
            # filter the noise samples
            if sample['content'] is not None:
                samples.append(sample)
            else:
                print("[Important warning] samples {} is ignored".format(sample))
        return samples

def load_test_data_samples(path_folder_base, test_id):
    data = []
    test = load_samples(f"{path_folder_base}/riteval_{test_id}.xml")
    for file_path in glob.glob(f"{path_folder_base}/riteval_{test_id}.xml"):
        data = data + load_samples(file_path)
    return data


def load_all_data_samples(path_folder_base):
    data = []
    for file_path in glob.glob("{}/*.xml".format(path_folder_base)):
        data = data + load_samples(file_path)
    return data

def check_false_labels(pred, false_labels):
	for label in false_labels:
		if label in pred:
			return True
	return False

from tqdm import tqdm

def format_output(text):
	CLEANR = re.compile('<.*?>') 
	cleantext = re.sub(CLEANR, '', text)
	return cleantext.strip().lower()

def readfile(filename):
    f = open(filename)
    data = json.load(f)
    return data

# datas = get_all_files_from_path("../data/COLIEE2024statute_data-English/fewshot")
from sentence_transformers import SentenceTransformer, util
query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')

def dpr(testfile, path="../data/COLIEE2024statute_data-English/fewshot"):
    datas = get_all_files_from_path(path)
    corpus = []
    content = []
    labels = []
    for data in datas:
        if testfile in data:
            continue
        data = load_samples(data)
        for item in data:
            # corpus.append(item["result"].replace("\n", " ").strip())
            corpus.append(item["result"].strip())
            content.append(item["content"].strip().replace(".", ""))
            labels.append(item["label"].strip())
    print(len(corpus))
    retrival_passage_embeddings = passage_encoder.encode(corpus)
    content_passage_embeddings = passage_encoder.encode(content)
    return corpus, content, labels, retrival_passage_embeddings, content_passage_embeddings

from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-xxl"
cache_dir = "/home/congnguyen/drive/.cache"
# cache_dir = ".cache"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(
		model_name, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16, load_in_8bit=True
	)

from tqdm import tqdm

def format_output(text):
	CLEANR = re.compile('<.*?>') 
	cleantext = re.sub(CLEANR, '', text)
	return cleantext.strip().lower()
    
def few_shot_prompting(indexes, corpus, content, labels, prompt_template):
    result = ""
    for i in indexes:
        answer = "Yes"
        if "N" == labels[i]:
            answer = "No"
        prompt = prompt_template.replace("{{premise}}", corpus[i]).replace('{{hypothesis}}', content[i]).replace('{{answer}}', answer)
        result += prompt
    return result
    
path_file = "../data/COLIEE2024statute_data-English/fewshot/"
list_prompt = readfile("../data/prompt4.json")
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

def predict(model, tokenizer, path_file, files=["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], output="../output/accuracy2/newpromt_"):
    for file in files:
        test_file = path_file+file+".xml"
        data = load_samples(test_file)
        corpus, content, labels, retrival_passage_embeddings, content_passage_embeddings = dpr(file)
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
                output_text = format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
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

testfiles = ["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"]
# testfiles = ["riteval_R04_en"]

predict(model, tokenizer, path_file, testfiles, "../output/fewshot_detail/")