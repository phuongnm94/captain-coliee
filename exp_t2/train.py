import os
import sys
import random
import argparse

from src.utils import load_json, save_json
from src.data import (
    build_task2_training_dataset,
    TrainDataset,
    TestDataset,
    NegativeSamplingCallback,
    MonoT5BatchCollator,
    BertBatchCollator,
)

import torch
import numpy as np

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from src.trainer import MonoT5Trainer


def train(config, save_dir, root_dir):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"
    print(device)

    seed = 442
    if config["negative_sampling_strategy"] != "random":
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    else:
        seed = np.random.randint(1, 100000)
    config["seed"] = seed

    train_dataset = TrainDataset(
        root_dir,
        config["dataset_path"],
        config["training_samples"],
        config["num_pairs_per_batch"],
        config["negative_sampling_strategy"],
        num_msmarco_pairs_per_batch=config.get("num_msmarco_pairs_per_batch", None),
        enhanced_weight=config.get("enhanced_weight", None),
        train_uncased=config["train_uncased"],
    )

    if "save_every_n_steps" in config:
        steps = config["save_every_n_steps"]
        strategy = "steps"
    else:
        steps = 1
        strategy = "epoch"
    ckpt_dir = os.path.join(save_dir, "ckpt")

    train_args = TrainingArguments(
        output_dir=ckpt_dir,
        do_train=True,
        save_strategy=strategy,
        save_steps=steps,
        logging_strategy=strategy,
        logging_steps=steps,
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        num_train_epochs=config["n_epochs"],
        warmup_ratio=config["warmup_ratio"],
        optim=config["optim"],
        label_smoothing_factor=config["label_smoothing_factor"],
        fp16=config["fp16"],
        seed=seed,
        disable_tqdm=False,
        load_best_model_at_end=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        deepspeed=config.get("deepspeed", None),
    )
    train_args.generation_config = None

    if config["model_class"] == "monot5":
        model = AutoModelForSeq2SeqLM.from_pretrained(config["base_model"]).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        collator_fn = MonoT5BatchCollator(tokenizer, device)

        trainer = MonoT5Trainer(
            loss_func=config["loss_fn"],
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=collator_fn,
        )

    elif config["model_class"] == "bert":
        model = AutoModelForSequenceClassification.from_pretrained(
            config["base_model"]
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        collator_fn = BertBatchCollator(tokenizer, device, max_length=512)

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=collator_fn,
        )

    else:
        raise ValueError(config["model_class"])

    ns_callback = NegativeSamplingCallback(trainer, config["model_class"])
    trainer.add_callback(ns_callback)

    trainer.train()

    trainer.save_model(ckpt_dir)
    trainer.save_state()

    save_json(os.path.join(save_dir, "train_configs.json"), config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("-s", "--save_dir", default=None, type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--root_dir", default="./", type=str
    )
    args = parser.parse_args()

    config = load_json(args.config_path)
    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(
            f"train_logs/{os.path.splitext(os.path.split(args.config_path)[1])[0]}"
        )
    train(config, save_dir, args.root_dir)


if __name__ == "__main__":
    main()
