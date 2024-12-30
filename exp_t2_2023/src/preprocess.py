from pathlib import Path
import os
import jsonlines
import subprocess
from tqdm import tqdm
import sys

root = Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(root))

from src.data import preprocess_case_data, get_task2_data
from src.utils import save_json
from src.eval_monot5 import predict_all_bm25

dataset_path = root/'data/task2_train_files_2024'

def create_bm25_indexes():
    tmp_dir = root / "data/bm25_indexes/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    for segment in ["train", "val", "test"]:
        
        indexes_dir = root/f'data/bm25_indexes/coliee_task2/{segment}'
        os.makedirs(indexes_dir, exist_ok=True)

        corpus_dir, cases_dir, _ = get_task2_data(dataset_path, segment=segment)

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

        subprocess.run(["python", "-m", "pyserini.index", "-collection", "JsonCollection",
                        "-generator", "DefaultLuceneDocumentGenerator", "-threads", "1", "-input",
                        f"{tmp_dir}", "-index", f"{indexes_dir}", "-storePositions", "-storeDocvectors",
                        "-storeRaw"])


def extract_negative_samples():
    bm25_index_path = str(root/'data/bm25_indexes/coliee_task2/train')

    _, cases_dir, label_data = get_task2_data(dataset_path, segment="train")
    bm25_scores = predict_all_bm25(dataset_path, bm25_index_path, eval_segment="train")

    num_negatives = 10
    sample_dict = {}
    for i, case in tqdm(enumerate(cases_dir)):
        bm25_score = bm25_scores[case]   
        top_negatives = sorted(bm25_score.items(), key=lambda x: x[1], reverse=True)[:num_negatives]
        negative_ids = [x[0] for x in top_negatives]
        sample_dict[case] = list(set(negative_ids + label_data[case]))
            
    save_path = root/'data/task2_training_negatives.json'
    save_json(save_path, sample_dict)


if __name__ == "__main__":
    create_bm25_indexes()
    extract_negative_samples()
    