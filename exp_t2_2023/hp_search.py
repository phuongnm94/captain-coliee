import os
import sys
import json
import time
import shutil
import subprocess

import optuna
import numpy as np

from eval_monot5 import eval_bm25_end_model


def evaluate_ckpts(model_dir, dataset_path, bm25_index_dir, segment="val",
                   eval_all_ckpts=True, config=None, model_class="monot5"):
    if eval_all_ckpts:
        ckpt_list = sorted([os.path.join(model_dir, f)
                            for f in os.listdir(model_dir) if f.startswith("checkpoint-")])
    else:
        ckpt_list = [model_dir]

    bm25_index_path = os.path.join(bm25_index_dir, segment)
    best_ckpt = None
    best_val, best_test, best_configs = [0, 0, 0], [0, 0, 0], None

    for ckpt in ckpt_list:
        print("\n==============================================================")
        print("Eval ckpt: ", ckpt)

        _best_val, _best_test, _best_configs = eval_bm25_end_model(
            dataset_path, bm25_index_path, ckpt, eval_segment=segment, model_class=model_class)
        if _best_test[0] > best_test[0]:
            best_val = _best_val
            best_test = _best_test
            best_configs = _best_configs
            best_ckpt = ckpt
    return best_val, best_test, best_ckpt, best_configs


ind = 1
with open(sys.argv[ind]) as f:
    configs = json.load(f)
config_name = os.path.splitext(os.path.split(sys.argv[ind])[1])[0]

n_trial = int(sys.argv[ind + 1])
tmp_dir = sys.argv[ind + 2]
val_segment = sys.argv[ind + 3]
root_dir = sys.argv[ind + 4]

best_ckpt, best_config, past_trials = [0, ""], [0, 0, 0], []
agg_val_res, agg_test_res = [], []

out_dir = "./train_logs/tuned"
best_model_dir = f"{out_dir}/{config_name}"
os.makedirs(best_model_dir, exist_ok=True)


def run_trial(trial):
    time.sleep(60)
    print(f"Start trial {trial.number}/{n_trial}")

    trial_configs = {}
    for k, v in configs.items():
        if isinstance(v, list):
            if len(v) == 2:
                if isinstance(v[0], float) or isinstance(v[1], float):
                    trial_configs[k] = trial.suggest_float(k, v[0], v[1])
                elif isinstance(v[0], int) and isinstance(v[1], int):
                    trial_configs[k] = trial.suggest_int(k, v[0], v[1])
                else:
                    trial_configs[k] = trial.suggest_categorical(k, v)
            else:
                trial_configs[k] = trial.suggest_categorical(k, list(set(v)))
        else:
            trial_configs[k] = v

    if trial_configs in past_trials:
        raise optuna.TrialPruned()
    print(trial_configs)
    past_trials.append(trial_configs)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    with open(f"{tmp_dir}/train_configs.json", "w") as f:
        json.dump(trial_configs, f)

    cmd = (f"{sys.executable} -u train.py {tmp_dir}/train_configs.json -s {tmp_dir} "
           f"--root_dir {root_dir} | tee -a {tmp_dir}/log")
    # cmd = (f"./scripts/tsz/run_training.sh {tmp_dir}/train_configs.json {tmp_dir}")
    try:
        p = subprocess.run(cmd, shell=True)
        if p.returncode != 0:
            raise optuna.TrialPruned()
    except Exception as e:
        print(e)
        raise optuna.TrialPruned()

    time.sleep(20)

    val_res, test_res, ckpt, conf = evaluate_ckpts(os.path.join(tmp_dir, "ckpt"),
                                     os.path.join(root_dir, configs["dataset_path"]),
                                     configs["bm25_index_dir"],
                                     segment=val_segment,
                                     model_class=trial_configs["model_class"]
                                    )
    agg_val_res.append(val_res)
    agg_test_res.append(test_res)
    if test_res[0] > best_ckpt[0]:
        best_ckpt[0] = test_res[0]
        best_ckpt[1] = os.path.split(ckpt)[1]
        for i in range(len(conf)):
            best_config[i] = conf[i]

    with open(f"{tmp_dir}/log", "a") as f:
        f.write(f"\n[Result] {val_res} - {test_res} - {conf} - {ckpt}")
    return test_res[0]


def save_best_model(study, trial):
    model_path = os.path.join(best_model_dir, str(trial.number))
    # if study.best_trial.number == trial.number:
    #     shutil.rmtree(best_model_dir, ignore_errors=True)
    #     shutil.move(tmp_dir, best_model_dir)
    if os.path.exists(tmp_dir):
        if os.path.exists(model_path):
            shutil.rmtree(model_path, ignore_errors=True)
        shutil.move(tmp_dir, model_path)


study = optuna.create_study(study_name="monot5_ft", direction="maximize")
study.optimize(run_trial, n_trials=n_trial, gc_after_trial=True,
               callbacks=[save_best_model])
print("Number of finished trials: ", len(study.trials))

df = study.trials_dataframe()
print(df)

trial = study.best_trial
print(" - Value: ", trial.value)
print(" - Params: ")
for key, value in trial.params.items():
    print("  - {}: {}".format(key, value))

print("Best trial: ", trial.number)
print("Best ckpt: ", best_ckpt)
print("Best configs: ", best_config)
print(f"Mean val: {np.mean(agg_val_res, axis=0)} - Std val: {np.std(agg_val_res, axis=0)}")
print(f"Mean test: {np.mean(agg_test_res, axis=0)} - Std val: {np.std(agg_test_res, axis=0)}")