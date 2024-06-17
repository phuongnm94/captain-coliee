#!/usr/bin/env python
import sys, os
os.environ["WANDB_DISABLED"] = "true"
sys.path.append("/path/pygaggle/")

import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
#import jsonlines
import argparse
from pygaggle.rerank.transformer import MonoT5
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

class MonoT5Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f'Query: {sample[0]} Document: {sample[1]} Relevant:'
        return {
          'text': text,
          'labels': sample[2],
        }

device = torch.device('cuda')
torch.manual_seed(123)

#model_name = './monoT5_model/monot5-large-msmarco'
model_name = './monoT5_model/monot5-large-msmarco-10k'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_df = pd.read_csv("./gen_df.csv")

train_df = pd.concat([train_df1, train_df2, train_df3, train_df4]).reset_index(drop=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df['label'] = np.where(train_df['label'] == 1, "true", "false")

train_samples = []
for i in tqdm(range(len(train_df))):
    q_id, question, a_id, context, label, _ = train_df.iloc[0]
    train_samples.append((question, context, label))


# In[8]:


# train_samples = []
# with open(args.triples_path, 'r', encoding="utf-8") as fIn:
#     for num, line in enumerate(fIn):
#         if num > 6.4e5 * args.epochs:
#             break
#         query, positive, negative = line.split("\t")
#         train_samples.append((query, positive, 'true'))
#         train_samples.append((query, negative, 'false'))

def smart_batching_collate_text_only(batch):
    texts = [example['text'] for example in batch]
    tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
    tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']

    for name in tokenized:
        tokenized[name] = tokenized[name].to(device)

    return tokenized

dataset_train = MonoT5Dataset(train_samples)
steps = 50
strategy = 'steps'

output_model_path = "monoT5_model/monot5_large_gen_from_model_60K"

logging_steps = 50
per_device_train_batch_size = 32
gradient_accumulation_steps = 16
learning_rate = 1e-7

# steps = 1
# strategy = 'epoch'

train_args = Seq2SeqTrainingArguments(
        output_dir=output_model_path,
        do_train=True,
        save_strategy=strategy,
        save_steps =steps, 
        logging_steps=logging_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=5e-5,
        num_train_epochs=2,
        warmup_steps=5,
        adafactor=True,
        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False,
    )

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=dataset_train,
    tokenizer=tokenizer,
    data_collator=smart_batching_collate_text_only,
)

trainer.train()
trainer.save_model(output_model_path)
trainer.save_state()

