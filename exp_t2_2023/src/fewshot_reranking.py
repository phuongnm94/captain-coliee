import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from pathlib import Path
import json
import random
from tqdm import tqdm
import sys
from ast import literal_eval

root = Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(root))

from src.utils import preprocess_case_data, format_output
from src.data import get_task2_data

def get_pred_cases(cases_dir, corpus_dir, top_k, margin, alpha):
    
    bm25_scores = json.load(open(root/'data/bm25_scores.json'))
    monot5_scores = json.load(open(root/'data/monot5_scores.json'))
    
    pred_cases = {}
    for case in cases_dir:
        bm25_score = bm25_scores[case]
        score = monot5_scores[case]

        candidate_dir = corpus_dir / case / "paragraphs"
        candidate_cases = sorted(os.listdir(candidate_dir))

        final_score = []
        for cand_case in candidate_cases:
            if alpha == 1:
                if cand_case not in bm25_score:
                    final_score.append([cand_case, 0.])
                else:
                    final_score.append([cand_case, score[cand_case]])
            else:
                final_score.append(
                    [
                        cand_case,
                        alpha * score[cand_case]
                        + (1 - alpha) * bm25_score.get(cand_case, 0),
                    ]
                )
        final_score = list(sorted(final_score, key=lambda x: -x[1]))

        top_ind = final_score[:top_k]
        pred_ind = [top_ind[0]]
        for cand in top_ind[1:]:
            if top_ind[0][1] - cand[1] < margin:
                pred_ind.append([cand[0], cand[1]])

        pred_cases[case] = pred_ind

    return pred_cases
        
def get_train_data():
    with open(root/'data/train_labels.json') as f:
        train_labels = json.load(f)
    with open(root/'data/val_labels.json') as f:
        val_labels = json.load(f)
        
    
    train_data = {}
    for case in (root/"data/task2_train_files_2024").iterdir():
        entailed_fragment = preprocess_case_data(case / "entailed_fragment.txt")
        candidates = []
        for cand in Path(case / "paragraphs").iterdir():
            cand_content = preprocess_case_data(cand)
            if int(case.name) <= 525:
                candidates.append(
                    [
                        cand.name,
                        cand_content,
                        1 if cand.name in train_labels[case.name] else 0,
                    ]
                )
            elif int(case.name) > 525 and int(case.name) <= 625:
                candidates.append(
                    [
                        cand.name,
                        cand_content,
                        1 if cand.name in val_labels[case.name] else 0,
                    ]
                )
        train_data[case.name] = {
            "fragment": entailed_fragment,
            "candidates": candidates,
        }
            
    return train_data

def zero_short_generate_prompt(query, candidates):
    zero_shot_prompt_template = "In bellow documents:\n{}\nQuestion: which documents really relevant to query '{}'?"
    
    document_map = [""] * len(candidates)
    candidate_string_list = []
    for i, cand in enumerate(candidates):
        candidate_string_list.append(f"Document {i+1}: {cand[1]}")
        document_map[i] = cand[0]
    prompt = zero_shot_prompt_template.format(
        "\n".join(candidate_string_list), query
    )
    return prompt, document_map

def shot_generate_prompt(query, candidates, num_doc_per_shot=5):
    shot_prompt_template = (
        'In bellow documents:\n{}\nThe documents really relevant to query "{}" '
    )
        
    document_map = []
    candidate_string_list = []
    positive_candidates = [cand for cand in candidates if cand[2] == 1]
    negative_candidates = [cand for cand in candidates if cand[2] == 0]
    candidates = positive_candidates + random.sample(
        negative_candidates,
        min(len(negative_candidates), num_doc_per_shot - len(positive_candidates)),
    )
    random.shuffle(candidates)
    for i, cand in enumerate(candidates):
        document_map.append(cand[0])
        candidate_string_list.append(f"Document {i+1}: {cand[1]}")

    prompt = shot_prompt_template.format("\n".join(candidate_string_list), query)

    answers = [document_map.index(cand[0]) for cand in positive_candidates]

    if len(answers) == 1:
        prompt += "is "
    else:
        prompt += "are "
    prompt += " ".join([f"document {a+1}" for a in answers]) + "."
    return prompt, document_map

def get_document_id(answer, document_map):
    return document_map[int(answer.split()[-1]) - 1]

def reranking(predicted_cases, train_data, llm_model, tokenizer):
    num_shots = 3
    final_preds = {}

    for case, predictions in tqdm(predicted_cases.items()):
        query = preprocess_case_data(
            root/f"data/task2_train_files_2024/{case}/entailed_fragment.txt"
        )
        candidates = [
            (
                pred[0],
                preprocess_case_data(
                    root/f"data/task2_train_files_2024/{case}/paragraphs/{pred[0]}"
                ),
            )
            for pred in predictions
        ]

        # Few-shot prompt generation
        samples = random.sample(list(train_data.items()), num_shots)
        few_shots = []
        for _, sample in samples:
            shot, document_map = shot_generate_prompt(
                sample["fragment"], sample["candidates"]
            )
            few_shots.append(shot) 
        last_shot, document_map = zero_short_generate_prompt(query, candidates)
        few_shots.append(last_shot)
        prompt = "\n####\n".join(few_shots)

        # LLM rerank
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt", padding="longest").to(llm_model.device)[
                "input_ids"
            ]
            outputs = llm_model.generate(inputs, max_new_tokens=2)
            raw_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [format_output(e.replace(prompt, "")) for e in raw_output]
            for text in output_text:
                try:
                    final_preds[case] = [get_document_id(text, document_map)]
                except:
                    print('\nParsing error')
                    final_preds[case] = [predictions[0][0]]
    return final_preds

def evaluate(pred_cases, label_data):
    tp = 0
    for case, pred in pred_cases.items():
        tp += len([p for p in pred if p[0] in label_data[case]])
    p = tp / sum([len(v) for _, v in pred_cases.items()])
    r = tp / sum([len(v) for _, v in label_data.items()])
    try:
        f = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f = None
    print(f, p, r)

if __name__ == '__main__':
    
    corpus_dir, cases_dir, label_data = get_task2_data(root/'data/task2_train_files_2024', segment='test')
    
    with open(root/'outputs/best_configs.txt') as f:
        configs = literal_eval(f.read().strip('\n'))
    print(f'Load best configs: {configs}')
    
    pred_cases = get_pred_cases(cases_dir, corpus_dir, configs[0], configs[1], configs[2])
    train_data = get_train_data()
    
    # Load Model
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    model_checkpoint = "google/flan-t5-xxl"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_checkpoint,
        device_map="auto"
    )
    
    pred_cases = {key: value for key, value in pred_cases.items() if int(key) < 630}
    label_data = {key: value for key, value in pred_cases.items()}
    final_preds = reranking(pred_cases, train_data, llm_model, tokenizer)
    
    output_folder = root/'outputs'
    os.makedirs(output_folder, exist_ok=True)
    with open(output_folder/'final_predictions.json', 'w') as f:
        f.write(json.dumps(final_preds))
    
    with open('./outputs/final_predictions.json') as f:
        final_preds = json.load(f)
        
    evaluate(final_preds, label_data)