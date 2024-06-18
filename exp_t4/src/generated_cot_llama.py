from os import listdir
from os.path import isfile, join

from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM,BitsAndBytesConfig,AutoModelForCausalLM
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
    return text+"\nAnswer: "+answer+"\nLet't think step by step why the answer is "+answer+"\n\nThe answer is "+answer+" because "

def write_cot(result, filename):
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in result]
    with open(filename, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line+'\n')




import copy

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
                label = item["label"]
                hypothesis = item["content"]
                premise = item["result"]
                #Important: You must use dot-product, not cosine_similarity
                text = prompting(premise, hypothesis, label, template_prompt)

                conversation=[]
                conversation.append({"role": "user", "content": text})
                
                input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
                # inputs = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
                outputs = model.generate(input_ids.cuda(), max_new_tokens=256)
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
#     predict(model, tokenizer, myfiles, "/home/s2320037/Collie/captain-coliee/exp_t4/output/generated_cot_by_llama2_2/")

import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--inference-file', type=str, required=True)
    parser.add_argument('--reference-file', type=str, required=True)
    args = parser.parse_args()

    path_file = "/home/s2320037/Collie/COLIEE2024statute_data-English/train/"
    myfiles = [f.replace(".xml", "") for f in listdir(path_file) if isfile(join(path_file, f))]
    myfiles.reverse()

    print(myfiles)
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    access_token = "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"
    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
    tokenizer = AutoTokenizer.from_pretrained(model_name,token=access_token,padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                token=access_token
            )

    list_prompt = ult.readfile("/home/s2320037/Collie/captain-coliee/exp_t4/data/prompt.json")
    predict(model, tokenizer, myfiles, "/home/s2320037/Collie/captain-coliee/exp_t4/output/generated_cot_by_llama2_2/")