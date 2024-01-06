import os
# import fire
import sys
sys.path.append('/home/s2210405/codes/coliee/24/llms_for_legal')

import random
import jsonlines
import subprocess

from tqdm import tqdm
from pathlib import Path

from src.data import preprocess_case_data, get_task2_data
from src.utils import load_json, save_json, save_txt
from eval_monot5 import predict_all_bm25, predict_all_monot5


def create_bm25_indexes(data_dir):
    # tmp_dir = "./data/bm25_indexes/tmp"
    tmp_dir = "/home/s2210405/codes/coliee/24/data/bm25_indexes/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    for segment in ["train", "val", "test"]:
        # indexes_dir = f"./data/bm25_indexes/coliee_task2/{data_dir}/{segment}"
        indexes_dir = f'/home/s2210405/codes/coliee/24/data/bm25_indexes/coliee_task2/{segment}'
        os.makedirs(indexes_dir, exist_ok=True)

        # dataset_path = Path(f"/home/s2210421/datasets/COLIEE2023/Task2/{data_dir}")
        dataset_path = Path(f'/home/s2210405/codes/coliee/24/data/task2_train_files_2024')
        corpus_dir, cases_dir, label_data = get_task2_data(dataset_path, segment=segment)

        for case in tqdm(cases_dir):
            candidate_dir = corpus_dir / case / "paragraphs"
            candidate_cases = sorted(os.listdir(candidate_dir))
            for cand_case in candidate_cases:
                cand_case_file = candidate_dir / cand_case
                cand_case_data = preprocess_case_data(cand_case_file)
                cand_num = cand_case.split(".txt")[0]
                dict_ = { "id": f"{case}_candidate{cand_num}.txt_task2", "contents": cand_case_data}

                with jsonlines.open(f"{tmp_dir}/candidate.jsonl", mode="a") as writer:
                    writer.write(dict_)

        # subprocess.run(["/home/s2210421/miniconda3/envs/dev/bin/python", "-m", "pyserini.index", "-collection", "JsonCollection",
        #                 "-generator", "DefaultLuceneDocumentGenerator", "-threads", "1", "-input",
        #                 f"{tmp_dir}", "-index", f"{indexes_dir}", "-storePositions", "-storeDocvectors",
        #                 "-storeRaw"])
        subprocess.run(["/home/s2210405/miniconda3/envs/coliee-24/bin/python", "-m", "pyserini.index", "-collection", "JsonCollection",
                        "-generator", "DefaultLuceneDocumentGenerator", "-threads", "1", "-input",
                        f"{tmp_dir}", "-index", f"{indexes_dir}", "-storePositions", "-storeDocvectors",
                        "-storeRaw"])


def extract_negative_samples():
    # dataset_path = "/home/thanhtc/mnt/datasets/COLIEE2023/Task2/data_org"
    # bm25_index_path = "./data/bm25_indexes/coliee_task2/data_org/train"
    dataset_path = '/home/s2210405/codes/coliee/24/data/task2_train_files_2024'
    bm25_index_path = '/home/s2210405/codes/coliee/24/data/bm25_indexes/coliee_task2/train'

    _, cases_dir, label_data = get_task2_data(dataset_path, segment="train")
    bm25_scores = predict_all_bm25(dataset_path, bm25_index_path, eval_segment="train")
    # monot5_scores = predict_all_monot5("./train_logs/tuned/monot5-large/ckpt", "train")

    num_negatives = 10
    sample_dict = {}
    for i, case in tqdm(enumerate(cases_dir)):
        bm25_score = bm25_scores[case]   
        top_negatives = sorted(bm25_score.items(), key=lambda x: x[1], reverse=True)[:num_negatives]
        negative_ids = [x[0] for x in top_negatives]

        sample_dict[case] = list(set(negative_ids + label_data[case]))
        if i == 0:
            print(case, sample_dict[case])

    # save_path = "/home/thanhtc/mnt/datasets/COLIEE2023/Task2/task2_training_negatives.json"
    save_path = '/home/s2210405/codes/coliee/24/data/task2_training_negatives.json'
    save_json(save_path, sample_dict)


if __name__ == "__main__":
    create_bm25_indexes(data_dir='/home/s2210405/codes/coliee/24/data')
    extract_negative_samples()
    