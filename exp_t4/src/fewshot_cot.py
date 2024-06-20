from os import listdir
from os.path import isfile, join
import os
from sentence_transformers import SentenceTransformer, util
import torch        
from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import json
from tqdm import tqdm
import utils.utils as ult



def readjsonl(filename="../output/generated_cot/prompt_0/riteval_R04_en_cot.jsonl"):
    jsonl_content = open(filename, "r", encoding="utf-8")
    result = [json.loads(jline) for jline in jsonl_content.read().splitlines()]
    return result
    
def load_cot_sample(testfile, path="../output/generated_cot/prompt_0"):
    files = ult.get_all_files_from_path(path)
    corpus = []
    prompts = []
    labels = []
    contents = []
    cots = []
    for file in files:
        if testfile in file:
            continue
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

# corpus, prompts, labels, contents, cots = load_cot_sample("/home/s2210436/Coliee2024/output/generated_cot/prompt_0")

def dpr(corpus):
    passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
    passage_embeddings = passage_encoder.encode(corpus)
    query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
    return passage_embeddings, query_encoder

def predict(model, tokenizer, path_file,fewshot_path, cot_path, list_prompt,files=["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], output="../output/accuracy2/newpromt_"):
    for file in files:
        # test_file = path_file+file+".xml"
        test_file = os.path.join(path_file,file+".xml")
        f = open(path_file+file+".txt", "w", encoding="utf-8")
        data = ult.load_samples(test_file)
        
        acc = {}
        for template_prompt in list_prompt:
            idx = template_prompt["id"]
            template_prompt = template_prompt["prompt"]
            corpus, prompts, labels, contents, cots = load_cot_sample(file, f"{cot_path}/prompt_"+str(idx))
            passage_embeddings, query_encoder = dpr(corpus)
            print(len(corpus))
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
                output_text = ult.format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
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



import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, required=False)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--list-prompt-path', type=str, required=False)
    parser.add_argument('--cot-path', type=str, required=False)
    parser.add_argument('--test-file-path', type=str, required=False)
    parser.add_argument('--output-file-path', type=str, required=False)
    parser.add_argument('--fewshot-path', type=str, required=True)
    args = parser.parse_args()

    
    # list_prompt = ult.readfile("/home/s2210436/Coliee2024/data/prompt.json") 
    # path_file = "/home/s2210436/Coliee2024/data/COLIEE2024statute_data-English/train/"
    list_prompt = ult.readfile(args.list_prompt_path)
    path_file = args.data_path

    # model_name = "google/flan-t5-xxl"
    # cache_dir = "/home/s2210436/.cache"
    model_name = args.model_name
    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto", cache_dir=cache_dir, torch_dtype=torch.float16, load_in_8bit=True
        )
        
    # testfiles = ["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"]
    if args.test_file_path is not None:
        testfiles = [args.test_file_path]
    else:
        testfiles = ["riteval_R04_en"]
    # cot_path = "/home/s2210436/Coliee2024/output/generated_cot"
    cot_path = args.cot_path
    # predict(model, tokenizer, path_file, cot_path, list_prompt,testfiles, "/home/s2210436/Coliee2024/output/fewshot_cot/newpromt_")
    predict(model, tokenizer, path_file,args.fewshot_path, cot_path, list_prompt,testfiles, args.output_file_path)