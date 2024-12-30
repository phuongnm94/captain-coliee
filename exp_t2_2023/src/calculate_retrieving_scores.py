from pathlib import Path
import json
import os
import sys

root = Path(os.path.realpath(__file__)).parents[1]
sys.path.insert(0, str(root))

from src.eval_monot5 import predict_all_bm25, predict_all_monot5

if __name__ == '__main__':
                
    bm25_scores = predict_all_bm25(
        dataset_path=root/"data/task2_train_files_2024",
        bm25_index_path=str(root/"data/bm25_indexes/coliee_task2/test"),
        eval_segment='test'
    )
    with open(root/'data/bm25_scores.json', 'w') as f:
        f.write(json.dumps(bm25_scores))
        
    monot5_scores = predict_all_monot5(
        ckpt_path=str(root/"train_logs/monot5-large-10k_hns/ckpt"),
        dataset_path=root/"data/task2_train_files_2024",
        eval_segment='test'
    )
    with open(root/'data/monot5_scores.json', 'w') as f:
        f.write(json.dumps(monot5_scores))