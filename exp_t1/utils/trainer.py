import os
import re
import copy
import random
import collections
import torch
import numpy as np
import pandas as pd
import json
import pickle
import nltk

from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (TrainerCallback, AutoTokenizer, Trainer, Seq2SeqTrainer, 
                          TrainingArguments, AutoModelForSeq2SeqLM)

class MonoT5BatchCollator:
    def __init__(self, tokenizer, device, max_length=512):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        self.pattern = '##Query: {} ##Document: {} ##Relevant:'
        # self.pattern = "Query: {} Document: {}"

    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def __call__(self, batch, return_tensors=None):
        texts = [self.pattern.format(example[0], self.tokenizer.decode(self.tokenizer.encode(example[1], 
                max_length=400, truncation=True), skip_special_tokens=True)) for b in batch for example in b]
        
        # with open('tmp.txt', 'a') as f:
        #     for t in texts:
        #         f.write(t + '\n\n')

        tokenized = self.tokenizer(texts, padding=True, truncation='longest_first',
                                   return_tensors='pt', max_length=self.max_length)
        tokenized['labels'] = self.tokenizer(
            ["true" if example[2] == 1 else "false"
            # [1 if example[2] == 1 else 0
             for b in batch for example in b], return_tensors='pt')['input_ids']
        tokenized["inst_w"] = torch.tensor(self.flatten([(1, example[3]) for b in batch for example in b]))
        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)
        return tokenized

class NegativeSamplingCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.trainer.train_dataset.create_training_dataset(self.trainer.model, int(state.epoch))


# BUILD TRAINER
class MonoT5Trainer(Seq2SeqTrainer):
    def __init__(self, loss_func, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder_input_ids = None
        self.token_false_id = self.tokenizer.get_vocab()["▁false"]
        self.token_true_id  = self.tokenizer.get_vocab()["▁true"]

        self.loss_func = loss_func

    def compute_loss(self, model, inputs, return_outputs=False):
        if "inst_w" in inputs.keys():
            inst_w = inputs.pop("inst_w")

        if self.decoder_input_ids is None:
            if isinstance(model, torch.nn.DataParallel) or \
                    isinstance(model, torch.nn.parallel.DistributedDataParallel):
                self.decoder_input_ids = model.module._shift_right(inputs["labels"])
            else:
                self.decoder_input_ids = model._shift_right(inputs["labels"])

        if self.loss_func == "cross_entropy":
            if isinstance(model, torch.nn.DataParallel) or \
                    isinstance(model, torch.nn.parallel.DistributedDataParallel):
                inputs["decoder_input_ids"] = model.module._shift_right(inputs["labels"])
            else:
                inputs["decoder_input_ids"] = model._shift_right(inputs["labels"])
            return super().compute_loss(model, inputs, return_outputs)
        
        elif self.loss_func in ["contrastive", "ensemble"]:
            xe_loss, logits = model(**inputs, use_cache=False)[:2]
            logits = logits[:, -1, [self.token_false_id, self.token_true_id]]
            scores = torch.nn.functional.log_softmax(logits, dim=1)
            log_probs = scores[:, 1]
            loss = torch.mean(
                -torch.log(torch.exp(log_probs[0]) / torch.sum(torch.exp(log_probs), dim=-1)))
            
        elif self.loss_func == "weighted_cross_entropy":
            xe_loss, logits = model(**inputs, use_cache=False)[:2]
            loss = inst_w * torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
                logits.view(-1, logits.size(-1)), inputs["labels"].view(-1))
            loss = torch.mean(loss)
        else:
            raise ValueError(self.loss_func)

        return loss

