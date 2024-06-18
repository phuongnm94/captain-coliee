#USE THIS

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
    return text+"\nLet't think step by step "


def write_cot(result, filename):
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in result]
    with open(filename, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line+'\n')



def predict(model, tokenizer,list_prompt, files=["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], output="../output/cot/newpromt_"):
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
                label = item["label"]
                hypothesis = item["content"]
                premise = item["result"]
                #Important: You must use dot-product, not cosine_similarity
                text = prompting(premise, hypothesis, label, template_prompt)
                inputs = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
                outputs = model.generate(inputs, max_new_tokens=256)
                output_text = ult.format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
                item.update({"prompt": text})
                item.update({"cot": output_text})
                result.append(item)
                if count < 3:
                    print(text)
                    print(output_text)
                    count += 1
            if not os.path.exists(output+f"prompt_{idx}"):
                os.makedirs(output+f"prompt_{idx}")
                write_cot(result, output+f"prompt_{idx}/"+file+f"_cot.jsonl")
# if __name__=="__main__":
#     predict(model, tokenizer, myfiles, "/home/s2210436/Coliee2024/output/cot/")
import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, required=False)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--list-prompt-path', type=str, required=False)
    parser.add_argument('--test-file-path', type=str, required=False)
    parser.add_argument('--output-cot-path', type=str, required=False)
    args = parser.parse_args()
    # list_prompt = ult.readfile("/home/s2210436/Coliee2024/data/prompt4.json")
    list_prompt = ult.readfile(args.list_prompt_path)
    # path_file = "/home/s2210436/Coliee2024/data/COLIEE2024statute_data-English/fewshot/"
    path_file = args.data_path
    myfiles = [f.replace(".xml", "") for f in listdir(path_file) if isfile(join(path_file, f))]
    print(myfiles)

    # model_name = "google/flan-t5-xxl"
    model_name = args.model_name

    # cache_dir = "/home/s2210436/.cache"
    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16, load_in_8bit=True
	)
    output_path = args.output_cot_path
    predict(model, tokenizer,list_prompt ,myfiles, output_path)