import torch
from tqdm import tqdm
from utils import load_samples, load_test_data_samples, load_all_data_samples
import re
from transformers import AutoTokenizer, BloomForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import json
from transformers import pipeline
import argparse
import csv
import os
import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	GenerationConfig,
	TextStreamer,
	pipeline,
)

parser = argparse.ArgumentParser('_')
parser.add_argument('--model_name', type=str, default="bigscience/mt0-xxl")
parser.add_argument('--setting_data', type=str, default="test")
parser.add_argument('--cache_dir', type=str, default="/home/congnguyen/drive/.cache")
parser.add_argument('--prompt_file', type=str, default="data/prompt.json")
parser.add_argument('--output_folder', type=str, default="results")
parser.add_argument('--float32', action='store_true')
parser.add_argument('--file', type=str, default="R04")
args = parser.parse_args()
print(args)

if not os.path.exists(args.output_folder + "/accuracy2"):
	os.makedirs(args.output_folder + "/accuracy2")

if not os.path.exists(args.output_folder + "/raw2"):
	os.makedirs(args.output_folder + "/raw2")

if args.float32:
	if "alpaca" in args.model_name:
		print(args.model_name)
		model = pipeline(model=args.model_name, device=0)
	else:
		tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
		model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)
		model = model.cuda()
else:
	print("Float16")
	if "alpaca" in args.model_name:
		model = pipeline(model=args.model_name, torch_dtype=torch.float16, device=0)
	else:
		tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
		model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=args.cache_dir)
		model = model.cuda()

# if "google/flan-t5-xxl" in args.model_name:
# 	tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
# 	model = AutoModelForSeq2SeqLM.from_pretrained(
# 		args.model_name, device_map="auto", cache_dir=args.cache_dir, torch_dtype=torch.float16, load_in_8bit=True
# 	)
# elif "alpaca" in args.model_name:
# 	tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
# 	model = AutoModelForSeq2SeqLM.from_pretrained(
# 		args.model_name, device_map="auto", cache_dir=args.cache_dir, torch_dtype=torch.float16, load_in_8bit=True
# 	)
# elif "llama2" in args.model_name:
# 	tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
# 	model = AutoModelForSeq2SeqLM.from_pretrained(
# 		args.model_name, device_map="auto", cache_dir=args.cache_dir, torch_dtype=torch.float16, load_in_8bit=True
# 	)
# elif "mistralai/Mixtral-8x7B" in args.model_name:
# 	print("Load "+args.model_name+" load_in_4bit ...!!!!")
# 	tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
# 	model = AutoModelForCausalLM.from_pretrained(
# 		args.model_name, device_map="auto", cache_dir=args.cache_dir, torch_dtype=torch.float16, load_in_4bit=True
# 	)
# elif "WizardLM/WizardLM-13B-V1.2" in args.model_name:
# 	print("Load "+args.model_name+" load_in_8bit ...!!!!")
# 	tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
# 	model = AutoModelForCausalLM.from_pretrained(
# 		args.model_name, device_map="auto", cache_dir=args.cache_dir, load_in_8bit=True
# 	)
# else:
# 	print("Load "+args.model_name+" load_in_8bit ...!!!!")
# 	tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
# 	model = AutoModelForCausalLM.from_pretrained(
# 		args.model_name, device_map="auto", cache_dir=args.cache_dir, torch_dtype=torch.float16, load_in_8bit=True
# 	)

def read_json_file(file_path):
	with open(file_path, 'r') as file:
		data = json.load(file)
	return data

def write_dict_to_json_file(data, file_path):
	with open(file_path, 'w') as file:
		json.dump(data, file, sort_keys=True, indent=4)

def write_to_csv_file(ids, predicted_labels, raw_predicted_labels, labels, file_path):
	with open(file_path, 'w', encoding="utf-8") as f:
		# create the csv writer
		writer = csv.writer(f)
		# write a row to the csv file
		writer.writerow(['id', 'predicted_label', 'raw_predicted_label', 'label'])
		for idx, predicted_label, raw_predicted_label, label in zip(ids, predicted_labels, raw_predicted_labels, labels):
			writer.writerow([idx, predicted_label, raw_predicted_label, label])

def format_output(text):
	cleantext = re.sub(CLEANR, '', text)
	return cleantext.strip().lower()

def check_false_labels(pred, false_labels):	
	for label in false_labels:
		if label in pred:
			return True
	return False

def check_labels(pred):	
	if "yes" in pred:
		return True
	return False
	

def format_prompt(prompt, system_prompt=""):
    if system_prompt.strip():
        return f"[INST] {system_prompt} {prompt} [/INST]"
    return f"[INST] {prompt} [/INST]"

def predict(context, query, prompt):
	text = prompt["prompt"].replace("{{premise}}", context).replace("{{hypothesis}}", query)
	if "mistral" in args.model_name:
		text = format_prompt(text)
	labels = prompt["label"]
	if "alpaca" in args.model_name:
		output_text = format_output(model(text, max_length=10, do_sample=False)[0]["generated_text"])
	else:
		inputs = tokenizer(text, return_tensors="pt")["input_ids"].cuda()
		outputs = model.generate(inputs, max_new_tokens=10)
		output_text = format_output(tokenizer.decode(outputs[0]).replace(text, "").split("\n")[-1])
	print(output_text)
	# if check_labels(output_text):
	# 	return "Y", output_text
	# else:
	# 	return "N", output_text
	if check_false_labels(output_text, false_labels):
		return "N", output_text
	else:
		return "Y", output_text

for en_jp in ["en"]:
	test_id = f"{args.file}_{en_jp}"
	if args.setting_data == "test":
		data = load_test_data_samples(f"data/COLIEE2024statute_data-English/train/", test_id)
	else:
		data = load_all_data_samples(f"data/COLIEE2024statute_data-English/train/")

	prompts = read_json_file(args.prompt_file)
	false_labels = set([prompt["label"][-1].lower() for prompt in prompts])
	CLEANR = re.compile('<.*?>') 
	results = {}
	for idx, prompt in enumerate(prompts):
		print("=================", f"Prompt {idx}", "=================")
		labels = []
		predicted_labels = []
		raw_predicted_labels = []
		ids = []
		for item in tqdm(data):
			label = item["label"]
			predicted_label, raw_predicted_label = predict(item["result"], item["content"], prompt)
			labels.append(label)
			print("predicted_label", predicted_label)
			print("gold", label)
			print("-"*20)
			predicted_labels.append(predicted_label)
			raw_predicted_labels.append(raw_predicted_label)
			ids.append(item["index"])
		
		# Write raw predictions
		raw_result_path = f"{args.output_folder}/raw/{args.model_name.split('/')[-1]}_results_{en_jp}_{args.setting_data}"
		if not os.path.exists(raw_result_path):
			os.makedirs(raw_result_path)
		write_to_csv_file(ids, predicted_labels, raw_predicted_labels, labels, raw_result_path + f"/prompt_{idx}.csv")

		acc = sum(1 for x,y in zip(labels, predicted_labels) if x == y) / len(predicted_labels)		
		results[f"Prompt {idx}"] = acc

	# Write accuracy results
	accuracy_result_path = f"{args.output_folder}/accuracy2/{args.model_name.split('/')[-1]}_results_{en_jp}_{args.setting_data}{args.file}.json"
	write_dict_to_json_file(results, accuracy_result_path)

import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--inference-file', type=str, required=True)
    parser.add_argument('--reference-file', type=str, required=True)
    args = parser.parse_args()