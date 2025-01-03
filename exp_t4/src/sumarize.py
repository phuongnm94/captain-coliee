from os import listdir
from os.path import isfile, join

from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import ujson
import utils.utils as ult
import os

from tqdm import tqdm


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




def predict(model, tokenizer, files=["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], output="../output/cot/newpromt_"):
    for file in files:
        test_file = path_file+file+".xml"
        print(test_file)
        data = ult.load_samples(test_file)
        
        acc = {}
        for prompt in list_prompt:
            template_prompt = prompt["prompt"]
            idx = prompt["id"]
            print(template_prompt)
            result = []
            count = 0
            for item in tqdm(data):
                label = "None"
                if "label" in item:
                    label = item["label"]
                hypothesis = item["content"]
                premise = item["result"].replace("\n", " ")
                #Important: You must use dot-product, not cosine_similarity
                text = prompting(premise, hypothesis, label, template_prompt)
                inputs = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
                outputs = model.generate(inputs, max_new_tokens=256)
                output_text = ult.format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
                item.update({"prompt": text})
                item.update({"sum": output_text})
                result.append(item)
                if count < 3:
                    print(text)
                    print(output_text)
                    count += 1
            if not os.path.exists(output+f"prompt_{idx}"):
                os.makedirs(output+f"prompt_{idx}")
            write_cot(result, output+f"prompt_{idx}/"+file+f"_sum.jsonl")
            

# if __name__=="__main__":
#     
import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--inference-file', type=str, required=True)
    parser.add_argument('--reference-file', type=str, required=True)
    args = parser.parse_args()

    path_file = "/home/s2210436/Coliee2024/data/COLIEE2024statute_data-English/fewshot/"
    myfiles = [f.replace(".xml", "") for f in listdir(path_file) if isfile(join(path_file, f))]
    print(myfiles)

    model_name = "google/flan-t5-xxl"
    cache_dir = "/home/s2210436/.cache"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16, load_in_8bit=True
        )
    list_prompt = ult.readfile("/home/s2210436/Coliee2024/data/prompt4.json")
    list_prompt[0]["prompt"] = 'Summarize this sentence: "{{premise}}"\nApproach: Issue, rule, application, conclusion.'
    predict(model, tokenizer, myfiles, "/home/s2210436/Coliee2024/output/sum/")

