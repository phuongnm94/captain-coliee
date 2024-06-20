from os import listdir
from os.path import isfile, join

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

def load_jsonl(file):
    with open(file) as f:
        data = [json.loads(line) for line in f]
    return data

from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-xxl"
cache_dir = ".cache"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(
		model_name, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16, load_in_8bit=True
	)

from tqdm import tqdm

def format_output(text):
	CLEANR = re.compile('<.*?>') 
	cleantext = re.sub(CLEANR, '', text)
	return cleantext.strip()#.lower()

def prompting(premise, hypothesis, label, template=None):
    if "true" in template.lower():
        answer = "True"
        if "N" in label:
            answer = "False"
    else:
        answer = "Yes"
        if "N" in label:
            answer = "No"
    text = template.replace("{{premise}}", premise).replace("{{hypothesis}}", hypothesis)
    return text

def write_cot(result, filename):
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in result]
    with open(filename, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line+'\n')

list_prompt = readfile("../data/prompt4.json")
list_prompt[0]["prompt"] = 'Summarize this sentence: "{{premise}}"\nApproach: Issue, rule, application, conclusion.'
list_prompt

import copy

def predict(model, tokenizer, files=["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], output="../output/cot/newpromt_"):
    for file in files:
        test_file = path_file+file+".xml"
        data = load_samples(test_file)
        
        acc = {}
        for template_prompt in list_prompt:
            template_prompt = template_prompt["prompt"]
            # print(template_prompt)
            result = []
            count = 0
            for item in tqdm(data):
                label = item["label"]
                hypothesis = item["content"]
                premise = item["result"]
                #Important: You must use dot-product, not cosine_similarity
                text = prompting(premise, hypothesis, label, template_prompt)
                inputs = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
                outputs = model.generate(inputs, max_new_tokens=256)
                output_text = format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
                item.update({"prompt": text})
                item.update({"sum": output_text})
                result.append(item)
                if count < 100:
                    print(text)
                    print(output_text)
                    count += 1
        #     acc.update({template_prompt: count/len(data)})
        # write_cot(result, output+file+"_cot.jsonl")

predict(model, tokenizer, ["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], "../output/generated_cot/")

from sentence_transformers import SentenceTransformer, util
query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')

def dpr(testfile=None, path="../data/COLIEE2024statute_data-English/fewshot"):
    datas = get_all_files_from_path(path)
    corpus = []
    content = []
    labels = []
    for data in datas:
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

corpus, content, labels, retrival_passage_embeddings, content_passage_embeddings = dpr()

template = "Document: {{premise}}\nQuestion: {{hypothesis}}? True or False "
path = "../output/sum/prompt_19"

files = get_all_files_from_path(path)
import torch
import json
import re
import random

full_data = []
for file in files:
    f = open(file, "r")
    data = [json.loads(line) for line in f]
    for item in data:
        count = 0
        premise = item["result"]
        old_hypo = item["content"]
        # print(item["content"])
        sum = item["sum"].replace("\"", "")
        sum = re.sub(r"\(\d\)", "", sum)
        
        if "The conclusion is that " in item["sum"]:
            sum = sum.split("The conclusion is that ")[-1].capitalize()
        elif "onclusion:" in item["sum"]:
            sum = sum.split("onclusion:")[-1].strip().capitalize()
        # elif "onclusion is:" in item["sum"]:
        #     sum = sum.split("onclusion is:")[-1].strip().capitalize()
        else:
            if len(sum.split(".")) > 1:
                sum = sum.split(".")[-2].strip().capitalize()+"."
        if ":" in sum:
            index = len(sum.split(":")[0])+1
            sum = sum[index:].strip().capitalize()
        if len(sum.split()) < 5:
            continue

        ##### sum
        result1 = {}
        index = item["index"]+"_aug_sum"
        result1.update({"id": index})
        # text = template.replace("{{premise}}", premise).replace("{{hypothesis}}", sum)
        # result1.update({"content": text})
        result1.update({"result": premise})
        result1.update({"content": sum})
        result1.update({"label": "Y"})
        full_data.append(result1)
        # sum + Yes
        if "_H" in file:
            result2 = {}
            if "Y" in item["label"]:
                label = "Y"
            else:
                label = "N"
                
            if label == "Y":
                result2.update({"id": index})
                index = item["index"]+"_aug_yes"
                # text = template.replace("{{premise}}", premise).replace("{{hypothesis}}", sum+" "+old_hypo)
                # result2.update({"content": text})
                result2.update({"result": premise})
                result2.update({"content": sum+" "+old_hypo})
                result2.update({"label": label})
                full_data.append(result2)
            # sum + No
            elif label == "N":
                index = item["index"]+"_aug_no"
                result2.update({"id": index})
                # text = template.replace("{{premise}}", premise).replace("{{hypothesis}}", sum+" "+old_hypo)
                # result2.update({"content": text})
                result2.update({"result": premise})
                result2.update({"content": sum+" "+old_hypo})
                result2.update({"label": "N"})
                full_data.append(result2)
                
        # sum + irrelavent
        query_embedding = query_encoder.encode(old_hypo)
        scores = util.dot_score(query_embedding, content_passage_embeddings)
        indexes = torch.topk(scores, 1200).indices[0]
        randid = random.randint(300, 1000)
        index = indexes[int(randid)]
        irr_hypo = content[index]
        # while(len(irr_hypo.split())>len(sum.split())):
        #     randid = random.randint(300, 1000)
        #     index = indexes[int(randid)]
        #     irr_hypo = content[index]
        result3 = {}
        index = item["index"]+"_aug_irr"
        result3.update({"id": index})
        # text = template.replace("{{premise}}", premise).replace("{{hypothesis}}", sum+ " "+irr_hypo)
        # result3.update({"content": text})
        result3.update({"result": premise})
        result3.update({"content": sum+ " "+irr_hypo})
        result3.update({"label": "N"})
        full_data.append(result3)

# f = open("../output/sum/full_data.jsonl", "w", encoding="utf-8")
import jsonlines
random.shuffle(full_data)
with jsonlines.open("../data/COLIEE2024statute_data-English/aug_sum/full_data.jsonl", mode='w') as writer:
    writer.write_all(full_data)


path_file = "../data/COLIEE2024statute_data-English/fewshot.json/"
files = get_all_files_from_path(path_file)
files

articles = {}

for file in files:
    data = load_jsonl(file)
    if "R05" in file: 
        for item in data:
            art = item["result"].split("\n")
            result = ""
            if art[0].split()[0] != "Article": print(art)
            for i in art:
                if i.split()[0] == "Article":
                    cur_art = " ".join(i.split()[0:2]).strip()
                    if result == "":
                        articles.update({cur_art: result.strip()})
                    else:
                        articles.update({pre_art: result.strip()})
                        result = ""
                    result = " ".join(i.split()[2:]).strip()+"\n"
                    pre_art = cur_art
                else:
                    result += i+"\n"
            if result != "":
                articles.update({pre_art: result.strip()})
        break
    else:
        for item in data:
            # print()
            art = item["result"].split("\n")
            result = ""
            if art[0].split()[0] != "Article": print(art)
            for i in art:
                if i.split()[0] == "Article":
                    cur_art = i.strip()
                    if result == "":
                        articles.update({cur_art: result.strip()})
                    else:
                        articles.update({pre_art: result.strip()})
                        result = ""
                    pre_art = cur_art
                else:
                    result += i+"\n"
            if result != "":
                articles.update({pre_art: result.strip()})

# articles


def write_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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
articles

model_name = "Qwen/Qwen-72B-Chat"
cache_dir = "/home/congnguyen/drive/.cache"
# cache_dir = ".cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

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
        write_json("../data/COLIEE2024statute_data-English/text/result_art.json", result_art)