import os

# import fire
import sys

sys.path.append("/home/s2210405/codes/coliee/24/llms_for_legal")

import random
import jsonlines
import subprocess

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from data import preprocess_case_data, get_task2_data
from utils import load_json, save_json, save_txt
from eval_monot5 import predict_all_bm25, predict_all_monot5

def parsea_args():
    parser = ArgumentParser()
    parser.add_argument("--tmp_dir", help="The temporal directory for data preprocessing", type=str, default="../tmp")
    parser.add_argument("--index_dir", help="The output directory for BM25 indexing process", type=str, default="../data/bm25_indices/coliee_task2")
    parser.add_argument("--segment", help="The segment data for BM25 indexing", type=str, choices=["train", "test"])
    parser.add_argument("--dataset_path", help="The path to the dataset for indexing", type=str, default="../data/coliee_task2")
    parser.add_argument("--neg_save_path", help="Saving path for negative samples", type=str, default="../data")
    args = parser.parse_args()
    return args

def create_bm25_indexes(args):
    tmp_dir = args.tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)
    segments = ["train", "val"] if args.segment == "train" else ["test"]
    for segment in segments:
        indexes_dir = f"{args.index_dir}/{segment}"
        os.makedirs(indexes_dir, exist_ok=True)

        dataset_path = Path(args.dataset_path)
        corpus_dir, cases_dir, _ = get_task2_data(dataset_path, segment=segment)

        for case in tqdm(cases_dir):
            candidate_dir = corpus_dir / case / "paragraphs"
            candidate_cases = sorted(os.listdir(candidate_dir))
            for cand_case in candidate_cases:
                cand_case_file = candidate_dir / cand_case
                cand_case_data = preprocess_case_data(cand_case_file)
                cand_num = cand_case.split(".txt")[0]
                dict_ = {
                    "id": f"{case}_candidate{cand_num}.txt_task2",
                    "contents": cand_case_data,
                }

                with jsonlines.open(f"{tmp_dir}/candidate.jsonl", mode="a") as writer:
                    writer.write(dict_)

        subprocess.run(
            [
                "../bin/python", # Your python directory
                "-m",
                "pyserini.index",
                "-collection",
                "JsonCollection",
                "-generator",
                "DefaultLuceneDocumentGenerator",
                "-threads",
                "1",
                "-input",
                f"{tmp_dir}",
                "-index",
                f"{indexes_dir}",
                "-storePositions",
                "-storeDocvectors",
                "-storeRaw",
            ]
        )


def extract_negative_samples(args):
    dataset_path = args.dataset_path
    bm25_index_path = (
        f"{args.index_dir}/train"
    )

    _, cases_dir, label_data = get_task2_data(dataset_path, segment="train")
    bm25_scores = predict_all_bm25(dataset_path, bm25_index_path, eval_segment="train")
    # monot5_scores = predict_all_monot5("./train_logs/tuned/monot5-large/ckpt", "train")

    num_negatives = 10
    sample_dict = {}
    for i, case in tqdm(enumerate(cases_dir)):
        bm25_score = bm25_scores[case]
        top_negatives = sorted(bm25_score.items(), key=lambda x: x[1], reverse=True)[
            :num_negatives
        ]
        negative_ids = [x[0] for x in top_negatives]

        sample_dict[case] = list(set(negative_ids + label_data[case]))
        if i == 0:
            print(case, sample_dict[case])

    save_path = "{}/task2_training_negatives_2024.json".format(args.neg_save_path)
    save_json(save_path, sample_dict)

def preprocess(args):
    create_bm25_indexes(args)
    extract_negative_samples(args)

if __name__ == "__main__":
    args = parsea_args()
    preprocess(args)
