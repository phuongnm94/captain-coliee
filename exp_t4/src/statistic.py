import os
import json
import statistics
file_path = "/home/s2320037/Collie/captain-coliee/exp_t4/"
output_dir = os.path.join(file_path,"output")
prompt_dir = os.path.join(file_path,"data")
prompt_file = os.path.join(prompt_dir,"prompt.json")
fs_dir = os.path.join(output_dir,"fewshot_cot")

results_file = [os.path.join(fs_dir,filename) for filename in os.listdir(fs_dir)]
with open(prompt_file,'r') as f:
    prompt_data = json.load(f)
# print(data)
# print(results_file)
new_analysis_path = os.path.join(output_dir,"analysis")
new_files = [os.path.join(new_analysis_path,filename) for filename in os.listdir(fs_dir)]
print(new_files)
results = {"prompts_performance":{},'variance':{},'mean':{}}
for index,filename in enumerate(results_file):
    with open(filename,'r') as f:
        data = json.load(f)
    new_dict = {}
    prefix = ""
    if "R01" in filename:
        prefix="R01"
    elif "R02" in filename:
        prefix="R02"
    elif "R03" in filename:
        prefix="R03"
    else:
        prefix="R04"
    for key in data:
        for prompt in prompt_data:
            if key == prompt["prompt"]:
                new_dict[prompt['id']] = data[key]
                if prompt['id'] not in results["prompts_performance"]:
                    results["prompts_performance"][prompt['id']] = [{prefix:data[key]}]
                else:
                    results["prompts_performance"][prompt['id']].append({prefix:data[key]})
    with open(new_files[index],'w') as f:
        json.dump(new_dict,f)
new_analysis_file = os.path.join(new_analysis_path,"variance.json")
for key in results['prompts_performance']:
    print(results['prompts_performance'][key])
    key_results = []
    for test in results['prompts_performance'][key]:
        for t_key in test:
            key_results.append(test[t_key])
    variance = statistics.variance(key_results)
    mean = statistics.mean(key_results)
    results['variance'][key] = variance
    results['mean'][key] = mean

with open(new_analysis_file,'w') as f:
    json.dump(results,f)

import argparse
if __name__ =="__main__":
    parser = argparse.ArgumentParser('_')
    parser.add_argument('--inference-file', type=str, required=True)
    parser.add_argument('--reference-file', type=str, required=True)
    args = parser.parse_args()