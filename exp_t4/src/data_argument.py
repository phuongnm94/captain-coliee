from os import listdir

from os.path import isfile, join
import utils.utils as ult
from bs4 import BeautifulSoup
import re
import json
from tqdm import tqdm
from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import torch
import random
import copy
import jsonlines
import argparse
import ujson





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
def collect_prompt():
    list_prompt = ult.readfile("../data/prompt.json")
    list_prompt[0]["prompt"] = 'Summarize this sentence: "{{premise}}"\nApproach: Issue, rule, application, conclusion.'
    return [list_prompt[0]]

# list_prompt


def predict(model, tokenizer, input_path,files=["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], output="../output/cot/newpromt_"):
    # path_file = "/home/s2320037/Collie/COLIEE2024statute_data-English/train/"
    for file in files:
        # test_file = input_path+file+".xml"
        test_file = join(input_path,file+".xml")
        data = ult.load_samples(test_file)
        list_prompt = collect_prompt()
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
                output_text = ult.format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
                item.update({"prompt": text})
                item.update({"sum": output_text})
                result.append(item)
                if count < 100:
                    print(text)
                    print(output_text)
                    count += 1
        #     acc.update({template_prompt: count/len(data)})
        write_cot(result, output+file+".jsonl")


from sentence_transformers import SentenceTransformer, util
query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
def summary_augument_data(args):
    def dpr(testfile=None, path="/home/s2320037/Collie/COLIEE2024statute_data-English/fewshot"):
        datas = ult.get_all_files_from_path(path)
        corpus = []
        content = []
        labels = []
        for data in datas:
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
    corpus, content, labels, retrival_passage_embeddings, content_passage_embeddings = dpr(path=args.xml_file_paths)
    summarize_model_name = args.summarize_model_name
    summarize_tokenizer = AutoTokenizer.from_pretrained(summarize_model_name)
    summarize_model = AutoModelForSeq2SeqLM.from_pretrained(
    		summarize_model_name, device_map="auto",torch_dtype=torch.float16, load_in_8bit=True
    	)
    # predict(summarize_model, summarize_tokenizer, ["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], "/home/s2320037/Collie/captain-coliee/exp_t4/output/sum/prompt1/")
    predict(summarize_model, summarize_tokenizer,args.xml_input_path, ["riteval_R01_en","riteval_R02_en","riteval_R03_en","riteval_R04_en"], args.summary_output_path)
    
    template = "Document: {{premise}}\nQuestion: {{hypothesis}}? True or False "
    # path = "/home/s2320037/Collie/captain-coliee/exp_t4/output/sum/prompt1/"

    files = ult.get_all_files_from_path(args.summary_output_path)

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
            indexes = torch.topk(scores, args.top_k).indices[0]
            randid = random.randint(300, 499)
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

        random.shuffle(full_data)
        with jsonlines.open(args.output_file_name, mode='w') as writer:
            writer.write_all(full_data)


    # path_file = "../data/COLIEE2024statute_data-English/fewshot.json/"
    # files = ult.get_all_files_from_path(path_file)

    # articles = {}

    # for file in files:
    #     data = ult.load_jsonl(file)
    #     if "R05" in file: 
    #         for item in data:
    #             art = item["result"].split("\n")
    #             result = ""
    #             if art[0].split()[0] != "Article": print(art)
    #             for i in art:
    #                 if i.split()[0] == "Article":
    #                     cur_art = " ".join(i.split()[0:2]).strip()
    #                     if result == "":
    #                         articles.update({cur_art: result.strip()})
    #                     else:
    #                         articles.update({pre_art: result.strip()})
    #                         result = ""
    #                     result = " ".join(i.split()[2:]).strip()+"\n"
    #                     pre_art = cur_art
    #                 else:
    #                     result += i+"\n"
    #             if result != "":
    #                 articles.update({pre_art: result.strip()})
    #         break
    #     else:
    #         for item in data:
    #             # print()
    #             art = item["result"].split("\n")
    #             result = ""
    #             if art[0].split()[0] != "Article": print(art)
    #             for i in art:
    #                 if i.split()[0] == "Article":
    #                     cur_art = i.strip()
    #                     if result == "":
    #                         articles.update({cur_art: result.strip()})
    #                     else:
    #                         articles.update({pre_art: result.strip()})
    #                         result = ""
    #                     pre_art = cur_art
    #                 else:
    #                     result += i+"\n"
    #             if result != "":
    #                 articles.update({pre_art: result.strip()})

# articles
def load_articles():
    articles_list = []
    articles = {}
    pre_art = ""
    result = ""
    print("we got in here") 
    with open("/home/s2320037/Collie/COLIEE2024statute_data-English/text/civil_code_en-1to724-2.txt", "r",encoding='utf-8') as f:
        for line in f:
            # print(line) 
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
        
    return articles, articles_list
# articles

def paraphrase(args):
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to('cuda')
    sentence = "This is something which i cannot understand at all"

    text =  "paraphrase: " + sentence + " </s>"

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")


    outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=256,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=5
    )

    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        print(line)
    
def main(args):
    # Sumarization
    summary_augument_data(args)
    # # extract rule
    # model_name = "Qwen/Qwen-72B-Chat"
    # # cache_dir = "/home/congnguyen/drive/.cache"
    # # cache_dir = ".cache"
    # # Note: The default behavior now has injection attack prevention off.
    # tokenizer = AutoTokenizer.from_pretrained(model_name,  trust_remote_code=True)
    # # device_map="auto", 
    # model = AutoModelForCausalLM.from_pretrained(model_name,  device_map="auto", 
    #                                             torch_dtype=torch.float16, trust_remote_code=True, load_in_4bit=True).eval()

    # prompt = "'{{text}}'. Analyze the structure following main premise, exception of each rule."
    # result_art = {}
    # # history = "Main Clause: A juridical person is not formed other than pursuant to the provisions of this Code or other laws.\nException Clauses:None"
    # articles, articles_list = load_articles()
    # # print(articles)
    # for art in articles:
    #     result_art.update({art: "Error OOM!"})
    #     text = prompt.replace("{{text}}", articles[art].strip())
    #     try:
    #         response, history = model.chat(tokenizer, text, history=None)
    #         print(response)
    #         result_art.update({art: response})
    #     except:
    #         ult.write_json("/home/s2320037/Collie/COLIEE2024statute_data-English/text/result_art.json", result_art)
        # return None
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    # parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--cache-dir', type=str, required=False)
    parser.add_argument('--data-path', type=str, required=False)
    parser.add_argument('--xml_file_paths', type=str, required=False)
    parser.add_argument('--xml_input_path', type=str, required=False)
    parser.add_argument('--summarize_model_name', type=str, required=False)
    parser.add_argument('--summary_output_path', type=str, required=False)
    # parser.add_argument('--list-prompt-path', type=str, required=False)
    # parser.add_argument('--cot-path', type=str, required=False)
    # parser.add_argument('--test-file-path', type=str, required=False)
    parser.add_argument('--output_file_name', type=str, required=False)
    parser.add_argument('--top_k', type=int, required=False)
    # parser.add_argument('--fewshot-path', type=str, required=False)
    args = parser.parse_args()
    main(args)