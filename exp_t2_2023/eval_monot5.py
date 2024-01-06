import os
import argparse
import itertools

from pathlib import Path
from collections import defaultdict
from src.utils import load_txt, segment_document, load_json, save_json

import torch
import optuna
import numpy as np

from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModel, \
    T5ForConditionalGeneration

from pyserini.search import LuceneSearcher
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

from src.data import get_task2_data, preprocess_case_data


def predict_bm25(searcher, doc, case):
    bm25_score = defaultdict(lambda: 0)
    hits = []
    segments = segment_document(doc, 1, 1)
    for segment in segments:
        _hits = searcher.search(segment[:1024], k=10000)
        hits.extend(_hits)

    for hit in hits:
        if hit.docid.endswith("task2"): 
            if hit.docid.split("_candidate")[0] == case:
                hit.docid = hit.docid.split("_task2")[0].split("_candidate")[1]
                bm25_score[hit.docid] = max(hit.score, bm25_score[hit.docid])
    return bm25_score


def predict_all_bm25(dataset_path, bm25_index_path, eval_segment="test",
                     k1=None, b=None, topk=None):
    searcher = LuceneSearcher(bm25_index_path)
    if k1 and b:
        print(f"k1: {k1}, b: {b}")
        searcher.set_bm25(k1, b)

    # dataset_path = "/home/thanhtc/mnt/datasets/COLIEE2023/Task2/data_org"
    corpus_dir, cases_dir, _ = get_task2_data(dataset_path, eval_segment)
    bm25_scores = {}
    for case in cases_dir:
        base_case_data = preprocess_case_data(corpus_dir / case / "entailed_fragment.txt")
        score = predict_bm25(searcher, base_case_data, case)
        if topk is not None:
            sorted_score = sorted(score.items(), key=lambda x: x[1], reverse=True)[:topk]
            score = {x[0]: x[1] for x in sorted_score}
        bm25_scores[case] = score
    return bm25_scores


def predict_monot5(reranker, doc, candidate_dir, config):
    if config["train_uncased"]:
        doc = doc.lower()
    query = Query(doc)
    texts = []

    candidate_cases = sorted(os.listdir(candidate_dir))
    for i, cand_case in enumerate(candidate_cases):
        cand_case_data = preprocess_case_data(
            candidate_dir / cand_case, uncased=config["train_uncased"])
        texts.append(Text(cand_case_data, metadata={"docid": cand_case}))

    monot5_score = defaultdict(lambda: 0)
    result = reranker.rerank(query, texts)
    for c in result:
        cand_id = c.metadata["docid"]
        monot5_score[cand_id] = max(
            monot5_score[cand_id], np.exp(c.score) * 100)
    return monot5_score


def predict_all_monot5(dataset_path, ckpt_path, eval_segment="test"):
    device = torch.device("cuda")
    model = T5ForConditionalGeneration.from_pretrained(ckpt_path).to(device).eval()
    reranker = MonoT5(model=model)

    if "ckpt" not in ckpt_path:
        config = {"train_uncased": False}
    else:
        root_dir = ckpt_path.split("ckpt")[0]
        config_path = os.path.join(root_dir, "train_configs.json")
        config = load_json(config_path)

    corpus_dir, cases_dir, _ = get_task2_data(dataset_path, eval_segment)
    monot5_scores = {}
    for case in cases_dir:
        base_case_data = preprocess_case_data(
            corpus_dir / case / "entailed_fragment.txt", uncased=config["train_uncased"])

        candidate_dir = corpus_dir / case / "paragraphs"
        score = predict_monot5(reranker, base_case_data, candidate_dir, config)
        monot5_scores[case] = score
    return monot5_scores


def eval_bm25(dataset_path, bm25_index_path, eval_segment="test"):
    corpus_dir, cases_dir, label_data = get_task2_data(dataset_path, eval_segment)

    bm25_score = predict_all_bm25(dataset_path, bm25_index_path, eval_segment)

    top_k = [1, 5, 10, 20, 50, 100]
    tp, fp, fn = [0] * len(top_k), [0] * len(top_k), [0] * len(top_k)
    for case in cases_dir:
        score = bm25_score[case]

        candidate_dir = corpus_dir / case / "paragraphs"
        candidate_cases = sorted(os.listdir(candidate_dir))

        label = [1 if f in label_data[case] else 0 for f in candidate_cases]
        final_score = [score[f] for f in candidate_cases]
        top_ind = np.argsort(final_score)
        for i, k in enumerate(top_k):
            pred_ind = [top_ind[-k:]]
            pred = np.zeros_like(label)
            pred[pred_ind] = 1

            tp[i] += np.sum([1 if a == b and a == 1 else 0 for a, b in zip(pred, label)])
            fp[i] += np.sum([1 if a != b and a == 1 else 0 for a, b in zip(pred, label)])
            fn[i] += np.sum([1 if a != b and a == 0 else 0 for a, b in zip(pred, label)])

    res = {}
    best_f1, best_p, best_r = 0, 0, 0
    for i, k in enumerate(top_k):
        p = tp[i] / (tp[i] + fp[i])
        r = tp[i] / (tp[i] + fn[i])
        f1 = 2 * ((p * r) / (p + r))
        if i == 0 and f1 > best_f1:
            best_f1 = f1
            best_p = p
            best_r = r

        res[f"F1@{k}"] = f1
        res[f"P@{k}"] = p
        res[f"R@{k}"] = r
        print(f"F1@{k}: {f1} - P@{k}: {p} - R@{k}: {r}")

    data_name = os.path.split(dataset_path)[1]
    output_dir = f"./results/coliee_task2/{data_name}/bm25"
    os.makedirs(output_dir, exist_ok=True)
    save_json(f"{output_dir}/{eval_segment}.json", res)
    return best_f1, best_p, best_r


def eval_bm25_end_model_ranking(dataset_path, bm25_scores, scores, top_k=1, margin=0, alpha=1,
                                eval_segment="test"):
    print(f"\n[{eval_segment}] k: {top_k} - margin: {margin} - alpha: {alpha}")
    corpus_dir, cases_dir, label_data = get_task2_data(dataset_path, eval_segment)

    tp, fp, fn = 0, 0, 0
    for case in cases_dir:
        bm25_score = bm25_scores[case]
        score = scores[case]

        candidate_dir = corpus_dir / case / "paragraphs"
        candidate_cases = sorted(os.listdir(candidate_dir))

        final_score = []
        for cand_case in candidate_cases:
            if alpha == 1:
                if cand_case not in bm25_score:
                    final_score.append(0)
                else:
                    final_score.append(score[cand_case])
            else:
                final_score.append(alpha * score[cand_case] + 
                                   (1 - alpha) * bm25_score.get(cand_case, 0))

        label = [1 if f in label_data[case] else 0 for f in candidate_cases]
        top_ind = np.argsort(final_score)[-top_k:]
        pred_ind = [top_ind[-1]]
        for i in top_ind[:-1]:
            if final_score[top_ind[-1]] - final_score[i] < margin:
                pred_ind.append(i)
        pred = np.zeros_like(label)
        pred[pred_ind] = 1

        tp += np.sum([1 if a == b and a == 1 else 0 for a, b in zip(pred, label)])
        fp += np.sum([1 if a != b and a == 1 else 0 for a, b in zip(pred, label)])
        fn += np.sum([1 if a != b and a == 0 else 0 for a, b in zip(pred, label)])

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * ((p * r) / (p + r))

    print(f"[{eval_segment}] Metrics: {[f1, p, r]} - {[top_k, margin, alpha]}")
    return [f1, p, r]


def eval_bm25_end_model(dataset_path, bm25_index_path, ckpt_path=None, top_k=None, margin=None, alpha=None,
                        eval_segment="test", model_class="monot5"):
    bm25_scores = predict_all_bm25(dataset_path, bm25_index_path, eval_segment)
    if model_class == "monot5":
        predict_func = predict_all_monot5
    else:
        raise ValueError(model_class)

    scores = predict_func(dataset_path, ckpt_path, eval_segment)
    if top_k is None:
        list_k = [1, 2, 3]
        list_margin = [0, 1, 2, 3, 4, 5]
        list_alpha = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

        best_metric = [0, 0, 0]
        best_config = []

        for k in list_k:
            for margin in list_margin:
                for alpha in list_alpha:
                    res = eval_bm25_end_model_ranking(
                        dataset_path, bm25_scores, scores, k, margin, alpha, eval_segment)
                    if res[0] > best_metric[0]:
                        best_metric = res
                        best_config = [k, margin, alpha]
        print(f"[{eval_segment}] Best metrics: {best_metric} - {best_config}")

        if eval_segment == "val":
            bm25_test_index_path = "./data/bm25_indexes/coliee_task2/data_org/test"
            bm25_test_scores = predict_all_bm25(dataset_path, bm25_test_index_path, eval_segment="test")
            test_scores = predict_func(dataset_path, ckpt_path, eval_segment="test")

            test_metric = eval_bm25_end_model_ranking(
                dataset_path, bm25_test_scores, test_scores,
                best_config[0], best_config[1], best_config[2], "test")
            return best_metric, test_metric, best_config

        return best_metric, best_config
    else:
        return eval_bm25_end_model_ranking(dataset_path, bm25_scores, scores, top_k, margin, alpha,
                                           eval_segment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["bm25", "monot5"])
    parser.add_argument("-m", "--model_dir", type=str, default="")
    parser.add_argument("-k", "--top_k", type=int, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--dataset_path", type=str, default="~/datasets/COLIEE2023/Task2")
    parser.add_argument("--bm25_index_path", type=str, default=None)
    parser.add_argument("--segment", choices=["val", "test"], type=str, default="test")
    parser.add_argument("--optuna", type=int, default=0)
    parser.add_argument("--pred_file", type=str, default=None)
    parser.add_argument("-s", "--save_path", type=str, default=None)

    args = parser.parse_args()
    if args.model == "bm25":
        eval_bm25(args.dataset_path, args.bm25_index_path, args.segment)
    elif args.model in ["monot5"]:
        eval_bm25_end_model(args.dataset_path, args.bm25_index_path, args.model_dir,
                            args.top_k, args.margin, args.alpha, args.segment, args.model)
    else:
        raise ValueError(args.model)
