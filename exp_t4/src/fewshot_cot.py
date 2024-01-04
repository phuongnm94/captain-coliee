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

import json

def readjsonl(filename="../output/generated_cot/prompt_0/riteval_R04_en_cot.jsonl"):
    jsonl_content = open(filename, "r", encoding="utf-8")
    result = [json.loads(jline) for jline in jsonl_content.read().splitlines()]
    return result
    
def load_cot_sample(path="../output/generated_cot/prompt_0"):
    files = get_all_files_from_path(path)
    corpus = []
    prompts = []
    labels = []
    contents = []
    cots = []
    for file in files:
        data = readjsonl(file)
        for item in data:
            corpus.append(item["result"])
            contents.append(item["content"])
            prompts.append(item["prompt"])
            cots.append(item["cot"].split(".")[0]+".")
            labels.append(item["label"])
    return corpus, prompts, labels, contents, cots


def fewshot_cot_prompting(indexes, prompts, labels, cots):
    result = ""
    for i in indexes:
        if "true or false" in prompts[i].lower():
            answer = "True"
            if "N" == labels[i]:
                answer = "False"
        else:
            answer = "Yes"
            if "N" == labels[i]:
                answer = "No"
        # prompt = prompt_template.replace("{{premise}}", corpus[i]).replace('{{hypothesis}}', content[i]).replace('{{answer}}', answer)
        prompt = prompting(prompts[i], cots[i], answer)
        result += prompt
    return result
    
def prompting(text, cot, answer):
    return text.split("\nAnswer: ")[0] + " Let's think step by step \nAnswer: "+ cot +" So the answer is " + answer+"\n\n"
    # return "### Instructs: "+text.split("\nAnswer: ")[0] + "\n### Response: "+ cot

corpus, prompts, labels, contents, cots = load_cot_sample("/home/s2210436/Coliee2024/output/generated_cot/prompt_0")

from sentence_transformers import SentenceTransformer, util
import torch
# Load the facebook-dpr model (pretrained model for English)
passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
passage_embeddings = passage_encoder.encode(corpus)
query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')

def predict(model, tokenizer, path_file, passage_embeddings, list_prompt, path_cots, files=["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], output="../output/accuracy2/newpromt_"):
    for file in files:
        test_file = path_file+file+".xml"
        f = open(path_file+file+".txt", "w", encoding="utf-8")
        data = load_samples(test_file)
        
        acc = {}
        for template_prompt in list_prompt:
            idx = template_prompt["id"]
            template_prompt = template_prompt["prompt"]
            corpus, prompts, labels, contents, cots = load_cot_sample(path_cots+"prompt_"+str(idx))
            result = []
            count = 0
            for item in tqdm(data):
                premise = item["result"]
                label = item["label"]
                hypothesis = item["content"]
                #Important: You must use dot-product, not cosine_similarity
                query_embedding = query_encoder.encode(premise)
                scores = util.dot_score(query_embedding, passage_embeddings)
                indexes = torch.topk(scores, 3).indices[0]
                prompt = fewshot_cot_prompting(indexes, prompts, labels, cots)
                if "true or false" in template_prompt.lower():
                    text = template_prompt.replace("{{premise}}", premise).replace("{{hypothesis}}", hypothesis) +"\nAnswer: True or False"
                else:
                    text = template_prompt.replace("{{premise}}", premise).replace("{{hypothesis}}", hypothesis) +"\nAnswer: Yes or No"
                text = prompt+text
                if count < 5:
                    print(text)
                    print("===================================")
                inputs = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
                outputs = model.generate(inputs, max_new_tokens=256)
                output_text = format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
                # print(output_text)
                # print("===================================")
                if "yes" in output_text or "true" in output_text:
                    output_text = "Y"
                else:
                    output_text = "N"
                # print("Predict: ", output_text, " Label: ", label)
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

def writefile(data, filename):
    # Serializing json
    json_object = json.dumps(data, indent=1)
    # Writing to sample.json
    with open(filename, "w") as outfile:
        outfile.write(json_object)
        
from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

list_prompt = readfile("/home/s2210436/Coliee2024/data/prompt.json") 
cots_path = "/home/s2210436/Coliee2024/output/generated_cot/"
path_file = "/home/s2210436/Coliee2024/data/COLIEE2024statute_data-English/train/"
model_name = "google/flan-t5-xxl"
cache_dir = "/home/s2210436/.cache"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(
		model_name, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16, load_in_8bit=True
	)
    
testfiles = ["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"]
# testfiles = ["riteval_R04_en"]
predict(model, tokenizer, path_file, passage_embeddings, testfiles, list_prompt, cots_path, "/home/s2210436/Coliee2024/output/fewshot_cot/newpromt_")
