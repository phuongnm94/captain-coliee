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
import json
def write_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

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
        write_json("../data/COLIEE2024statute_data-English/text/result_art_fujisan.json", result_art)