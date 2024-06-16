import os
from os import listdir
from os.path import isfile, join
import jsonlines

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

def xml2json(files, out_path):
    result = []
    for file in files:
        data = load_samples(file)
        for k in data:
            item = {}
            if 'index' not in k:
                print(k)
            item.update({"id": k['index']})
            item.update({"content": template.replace("{{premise}}", k["result"]).replace("{{hypothesis}}", k["content"])})
            if k["label"] == "Y": item.update({"label": "true"})
            else: item.update({"label": "false"})
            result.append(item)

    if "R0" in files[0] or "H30" in files[0]:
        out_path = out_path+file.split("/")[-1].replace(".xml", "")+".jsonl"
    with jsonlines.open(out_path, 'w') as writer:
        writer.write_all(result)

def loadjsonl(files, out_path):
    result = []
    for file in files:
        f = readfile(file)
        item = {}
        for line in f:
            data = line
            item.update({"id": data['index']})
            item.update({"content": data['prompt']})
            item.update({"label": data['label']})
            result.append(item)

    with jsonlines.open(out_path, 'w') as writer:
        writer.write_all(result)

#################################
template = "Document: {{premise}}\nQuestion: {{hypothesis}}? True or False "

strategy = "zeroshot_aug_r32"
batch_size = 32
prompt_id = 20
epochs = 100
output_data = f"../output/finetuned/data/{strategy}_lora-flan-t5-large_prompt{prompt_id}_batch{str(batch_size)}_epochs{str(epochs)}"

if not os.path.exists(output_data):
	os.makedirs(output_data)

if "zeroshot" in strategy:
    import shutil
    new_path = shutil.copy(f"../data/COLIEE2024statute_data-English/train_full.jsonl", f"{output_data}/train_full.jsonl")
    train_path = get_all_files_from_path("../data/COLIEE2024statute_data-English/train")
    out_trainpath = f"{output_data}/train.jsonl"

    test_path = get_all_files_from_path("../data/COLIEE2024statute_data-English/test")
    out_testpath = f"{output_data}/"
    xml2json(train_path, out_trainpath)
    for i in range(len(test_path)):
        xml2json([test_path[i]], out_testpath)
elif "fewshot" in strategy:
    train_path = get_all_files_from_path(f"../data/finetune_exp/{strategy}/")
    out_trainpath = f"{output_data}/train.jsonl"
    loadjsonl(train_path, out_trainpath)

    test_path = get_all_files_from_path("../data/COLIEE2024statute_data-English/test")
    out_testpath = f"{output_data}/"
    xml2json([test_path[0]], out_testpath)
    xml2json([test_path[1]], out_testpath)
    xml2json([test_path[2]], out_testpath)
    xml2json([test_path[3]], out_testpath)


from datasets import load_dataset

dataset = load_dataset("json", data_files={"train": f"{output_data}/train.jsonl", 
                                          "test1": f"{output_data}/riteval_R01_en.jsonl",
                                          "test2": f"{output_data}/riteval_R02_en.jsonl",
                                          "test3": f"{output_data}/riteval_R03_en.jsonl",
                                          "test4": f"{output_data}/riteval_R04_en.jsonl"})

print(dataset)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id="google/flan-t5-large"

# Load tokenizer of FLAN-t5-XL
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/home/congnguyen/drive/.cache")

from datasets import concatenate_datasets
import numpy as np
# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test4"]]).map(lambda x: tokenizer(x["content"], truncation=True), batched=True, remove_columns=["content", "label"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 99))
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test4"]]).map(lambda x: tokenizer(x["label"], truncation=True), batched=True, remove_columns=["content", "label"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 5))
print(f"Max target length: {max_target_length}")

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["Classification: " + item for item in sample["content"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["label"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["content", "label", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")
# save datasets to disk for later easy loading
tokenized_dataset["train"].save_to_disk(f"../output/finetuned/data/{output_data}/train")
tokenized_dataset["test4"].save_to_disk(f"../output/finetuned/data/{output_data}/eval")

from transformers import AutoModelForSeq2SeqLM

# huggingface hub model id
# model_id = "philschmid/flan-t5-large-sharded-fp16"
model_id="google/flan-t5-large"
# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir="/home/congnguyen/drive/.cache")

# from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

# # Define LoRA Config
# lora_config = LoraConfig(
#  r=16,
#  lora_alpha=32,
#  target_modules=["q", "v"],
#  lora_dropout=0.05,
#  bias="none",
#  task_type=TaskType.SEQ_2_SEQ_LM
# )
# # prepare int-8 model for training
# model = prepare_model_for_int8_training(model)

# # add LoRA adaptor
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

import os
os.environ["WANDB_PROJECT"] = "finetuned-flan-t5-large"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

output_dir=f"../output/finetuned/{strategy}_lora-flan-t5-large_prompt{prompt_id}_batch{str(batch_size)}_epochs{str(epochs)}"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
	# auto_find_batch_size=True,
    per_device_train_batch_size=batch_size,
    learning_rate=3e-5, # higher learning rate
    num_train_epochs=epochs,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_total_limit=2,
    save_strategy="steps",
    report_to="wandb",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test4"]
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# train model
trainer.train()

# Save our LoRA model & tokenizer results
peft_model_id=f"../output/finetuned/{strategy}_peft_results_prompt{prompt_id}_batch{str(batch_size)}_epochs{str(epochs)}"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
# if you want to save the base model to call
# trainer.model.base_model.save_pretrained(peft_model_id)