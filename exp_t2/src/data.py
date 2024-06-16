import os
import re
import copy
import random
import collections

from src.utils import load_txt, load_json, filter_document
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5, MonoBERT

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from transformers import TrainerCallback, AutoTokenizer


def flatten(l):
    return [item for sublist in l for item in sublist]


def preprocess_case_data(
    file_path,
    max_length=None,
    min_sentence_length=None,
    uncased=False,
    filter_min_length=None,
):
    if not os.path.exists(file_path):
        return None

    text = load_txt(file_path)

    text = (
        text.strip()
        .replace("\n", " ")
        .replace("FRAGMENT_SUPPRESSED", "")
        .replace("FACTUAL", "")
        .replace("BACKGROUND", "")
        .replace("ORDER", "")
    )
    if uncased:
        text = text.lower()
    text = re.sub("\s+", " ", text).strip()
    text = " ".join([w for w in text.split() if w])

    cite_number = re.search("\[[0-9]+\]", text)
    if cite_number:
        text = text[cite_number.span()[1] :].strip()
    if filter_min_length:
        words = text.split()
        if len(words) <= filter_min_length:
            return None

    if min_sentence_length:
        text = filter_document(text, min_sentence_length)
    if max_length:
        words = text.split()[:max_length]
        text = " ".join(words)
    if not text.endswith("."):
        text = text + "."
    return text


def get_task2_data(dataset_path, segment="test"):
    if segment == "train":
        start_idx, end_idx = 0, 625
    elif segment == "val":
        start_idx, end_idx = 625, 725
    elif segment == "test":
        start_idx, end_idx = 725, 825
    else:
        start_idx, end_idx = 0, 1000

    corpus_dir = Path(dataset_path)
    root_dir = corpus_dir.parent
    if segment == "train":
        label_data = load_json(root_dir / "train_labels_2024.json")
    elif segment == "val":
        label_data = load_json(root_dir / "val_labels_2024.json")
    elif segment == "test":
        # label_data = load_json(root_dir / "test_labels_2024.json")
        label_data = {}
    else:
        label_data = load_json(root_dir / "train_labels.json")
        label_data.update(load_json(root_dir / "val_labels.json"))
        label_data.update(load_json(root_dir / "test_labels.json"))
    cases_dir = sorted(os.listdir(corpus_dir))
    return corpus_dir, cases_dir[start_idx:end_idx], label_data


def get_msmarco_dataset(root_dir):
    dataset_path = os.path.join(root_dir, "datasets/triples/triples.train.small.tsv")

    train_samples = []
    with open(dataset_path, "r", encoding="utf-8") as fIn:
        for num, line in enumerate(fIn):
            if num > 6.4e5:
                break
            query, positive, negative = line.split("\t")
            train_samples.append([(query, positive, 1), (query, negative, 0)])
    return train_samples


def build_task2_training_dataset(
    dataset_path, training_samples_file, train_uncased=False
):
    corpus_dir, cases_dir, label_data = get_task2_data(dataset_path, segment="train")
    training_samples = {}
    if training_samples_file:
        training_samples = load_json(training_samples_file)

    dataset = []
    for case in cases_dir:
        base_case_file = corpus_dir / case / "entailed_fragment.txt"
        base_case_data = preprocess_case_data(base_case_file, uncased=train_uncased)
        label = label_data[case]

        case_dict = {
            "id": case,
            "text": base_case_data,
            "pos_candidates": [],
            "neg_candidates": [],
        }

        candidate_dir = corpus_dir / case / "paragraphs"
        candidate_cases = sorted(os.listdir(candidate_dir))
        for cand_case in candidate_cases:
            if case in training_samples and cand_case not in training_samples[case]:
                continue
            cand_case_file = candidate_dir / cand_case
            cand_case_data = preprocess_case_data(
                cand_case_file, uncased=train_uncased, filter_min_length=10
            )
            if cand_case_data is None:
                continue

            l = "pos_candidates" if cand_case in label else "neg_candidates"
            case_dict[l].append({"id": cand_case, "text": cand_case_data})
        dataset.append(case_dict)
    return dataset


class TrainDataset(Dataset):
    def __init__(
        self,
        root_dir,
        dataset_path,
        training_samples_file,
        num_pairs_per_batch,
        ns_strategy,
        num_msmarco_pairs_per_batch=None,
        enhanced_weight=None,
        train_uncased=False,
    ):
        self.data = build_task2_training_dataset(
            os.path.join(root_dir, dataset_path), training_samples_file, train_uncased
        )
        self.ns_strategy = ns_strategy

        self.training_data = []
        self.ps = {
            sample["id"]: [pos["id"] for pos in sample["pos_candidates"]]
            for sample in self.data
        }
        self.ps_iter = copy.deepcopy(self.ps)
        self.ns = {
            sample["id"]: [neg["id"] for neg in sample["neg_candidates"]]
            for sample in self.data
        }
        self.ns_iter = copy.deepcopy(self.ns)

        self.use_msmarco_dataset = num_msmarco_pairs_per_batch is not None
        if self.use_msmarco_dataset:
            self.ms_training_data = get_msmarco_dataset(root_dir)
            self.num_msmarco_pairs_per_batch = num_msmarco_pairs_per_batch
            self.num_pairs_per_batch = num_pairs_per_batch - num_msmarco_pairs_per_batch
        else:
            self.num_pairs_per_batch = num_pairs_per_batch
        self.use_instance_weighting = enhanced_weight is not None
        self.enhanced_weight = enhanced_weight

    def __len__(self):
        return len(self.data)

    def create_training_dataset(self, model, epoch, model_class):
        np.random.shuffle(self.data)
        if self.ns_strategy == "random":
            self.random_negative_sampling_dataset(model)
        elif self.ns_strategy == "hard":
            self.hard_negative_sampling_dataset(model, model_class)
        elif self.ns_strategy == "seq":
            self.seq_negative_sampling_dataset(model_class)
        elif self.ns_strategy == "random_hard":
            if epoch % 2 == 0:
                self.random_negative_sampling_training_dataset()
            else:
                self.hard_negative_sampling_training_dataset(model)

    def random_negative_sampling_dataset(self, model):
        model.eval()

        self.training_data = []
        for sample in self.data:
            batch = []

            for cand in sample["pos_candidates"]:
                batch.append((sample["text"], cand["text"], 1, 1))

            num_neg_pairs = max(
                self.num_pairs_per_batch - len(sample["pos_candidates"]), 0
            )
            if len(sample["neg_candidates"]):
                neg_cands_id = np.random.choice(
                    len(sample["neg_candidates"]), num_neg_pairs, replace=True
                )
                neg_cands = [sample["neg_candidates"][i] for i in neg_cands_id]
                for cand in neg_cands:
                    batch.append((sample["text"], cand["text"], 0, 1))

            if self.use_msmarco_dataset:
                ms_data_ind = np.random.choice(
                    len(self.ms_training_data), self.num_msmarco_pairs_per_batch
                )
                for i in ms_data_ind:
                    batch.append((*self.ms_training_data[i][0], 1))
                    batch.append((*self.ms_training_data[i][1], 1))
            self.training_data.append(batch)

        model.train()

    def hard_negative_sampling_dataset(self, model, model_class):
        model.eval()
        if model_class == "monot5":
            reranker = MonoT5(model=model)
        elif model_class == "bert":
            reranker = MonoBERT(model=model)
        else:
            raise ValueError(model_class)

        self.training_data = []
        for sample in tqdm(self.data):
            batch = []

            texts = []
            for i, cand in enumerate(sample["neg_candidates"]):
                texts.append(Text(cand["text"], metadata={"id": cand["id"]}))
            if self.use_instance_weighting:
                for i, cand in enumerate(sample["pos_candidates"]):
                    texts.append(Text(cand["text"], metadata={"id": cand["id"]}))

            query = Query(sample["text"])
            results = reranker.rerank(query, texts)
            if self.use_instance_weighting:
                pos_ids = [cand["id"] for cand in sample["pos_candidates"]]
                neg_ids = [cand["id"] for cand in sample["neg_candidates"]]

                top_idxs = [c.metadata["id"] for c in results][
                    : self.num_pairs_per_batch
                ]
                last_pos_idx, top_neg_idx = -1, 100
                for i, idx in enumerate(top_idxs):
                    if idx in pos_ids:
                        last_pos_idx = i
                    if idx in neg_ids and top_neg_idx == 100:
                        top_neg_idx = i

                for i, cand in enumerate(sample["pos_candidates"]):
                    if cand["id"] not in top_idxs or (
                        cand["id"] in top_idxs
                        and top_idxs.index(cand["id"]) > top_neg_idx
                    ):
                        cand_w = self.enhanced_weight
                    else:
                        cand_w = 1
                    batch.append((sample["text"], cand["text"], 1, cand_w))

                num_neg_pairs = max(
                    self.num_pairs_per_batch - len(sample["pos_candidates"]), 0
                )
                neg_cands = [
                    cand for cand in sample["neg_candidates"] if cand["id"] in top_idxs
                ]
                for i, cand in enumerate(neg_cands):
                    if top_idxs.index(cand["id"]) < last_pos_idx:
                        cand_w = self.enhanced_weight
                    else:
                        cand_w = 1
                    batch.append((sample["text"], cand["text"], 0, cand_w))
            else:
                for i, cand in enumerate(sample["pos_candidates"]):
                    batch.append((sample["text"], cand["text"], 1, 1))

                num_neg_pairs = max(
                    self.num_pairs_per_batch - len(sample["pos_candidates"]), 0
                )
                top_neg_idxs = [c.metadata["id"] for c in results][:num_neg_pairs]
                neg_cands = [
                    cand
                    for cand in sample["neg_candidates"]
                    if cand["id"] in top_neg_idxs
                ]
                for i, cand in enumerate(neg_cands):
                    batch.append((sample["text"], cand["text"], 0, 1))

            if self.use_msmarco_dataset:
                ms_data_ind = np.random.choice(
                    len(self.ms_training_data), self.num_msmarco_pairs_per_batch
                )
                for i in ms_data_ind:
                    batch.append((*self.ms_training_data[i][0], 1))
                    batch.append((*self.ms_training_data[i][1], 1))
            self.training_data.append(batch)

        model.train()

    def seq_negative_sampling_dataset(self):
        self.training_data = []

        for sample in self.data:
            batch = []
            if len(self.ps[sample["id"]]) == 1:
                pos_cands = [sample["pos_candidates"][0]["text"]]
            else:
                if len(self.ps_iter[sample["id"]]) > 1:
                    pos_cand_id = copy.deepcopy(self.ps_iter[sample["id"]][:1])
                    self.ps_iter[sample["id"]] = self.ps_iter[sample["id"]][1:]
                else:
                    diff = 1 - len(self.ps_iter[sample["id"]])
                    pos_cand_id = copy.deepcopy(self.ps_iter[sample["id"]])
                    pos_cand_id += copy.deepcopy(self.ps[sample["id"]][:diff])
                    self.ps_iter[sample["id"]] = copy.deepcopy(
                        self.ps[sample["id"]][diff:]
                    )
                pos_cands = [
                    pos["text"]
                    for pos in sample["pos_candidates"]
                    if pos["id"] in pos_cand_id
                ]
            for t in pos_cands:
                batch.append((sample["text"], t, 1))

            n_neg_sample = min(self.ns_ratio, len(sample["neg_candidates"]))
            if len(self.ns_iter[sample["id"]]) > n_neg_sample:
                neg_cands_id = copy.deepcopy(self.ns_iter[sample["id"]][:n_neg_sample])
                self.ns_iter[sample["id"]] = self.ns_iter[sample["id"]][n_neg_sample:]
            else:
                diff = n_neg_sample - len(self.ns_iter[sample["id"]])
                neg_cands_id = copy.deepcopy(self.ns_iter[sample["id"]])
                neg_cands_id += copy.deepcopy(self.ns[sample["id"]][:diff])
                self.ns_iter[sample["id"]] = copy.deepcopy(self.ns[sample["id"]][diff:])
            neg_cands = [
                neg["text"]
                for neg in sample["neg_candidates"]
                if neg["id"] in neg_cands_id
            ]
            for t in neg_cands:
                batch.append((sample["text"], t, 0))

            self.training_data.append(batch)

    def __getitem__(self, idx):
        return self.training_data[idx]


class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data

        self.test_data = []


class MonoT5BatchCollator:
    def __init__(self, tokenizer, device, max_length=512):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

        self.pattern = "Query: {} Document: {} Relevant:"
        # self.pattern = "Query: {} Document: {}"

    def __call__(self, batch, return_tensors=None):
        texts = [
            self.pattern.format(example[0], example[1]) for b in batch for example in b
        ]
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_length,
        )
        tokenized["labels"] = self.tokenizer(
            [
                "true" if example[2] == 1 else "false"
                # [1 if example[2] == 1 else 0
                for b in batch
                for example in b
            ],
            return_tensors="pt",
        )["input_ids"]
        tokenized["inst_w"] = torch.tensor(
            flatten([(1, example[3]) for b in batch for example in b])
        )
        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)
        return tokenized


class BertBatchCollator:
    def __init__(self, tokenizer, device, max_length=256):
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def __call__(self, batch, return_tensors=None):
        texts = [(example[0], example[1]) for b in batch for example in b]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_length,
        )
        tokenized["labels"] = torch.tensor([example[2] for b in batch for example in b])
        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.device)
        return tokenized


class NegativeSamplingCallback(TrainerCallback):
    def __init__(self, trainer, model_class="monot5"):
        self.trainer = trainer
        self.model_class = model_class

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.trainer.train_dataset.create_training_dataset(
            self.trainer.model, int(state.epoch), self.model_class
        )
