from bs4 import BeautifulSoup
import argparse
import csv
import os

parser = argparse.ArgumentParser('_')
parser.add_argument('--inference-file', type=str, required=True)
parser.add_argument('--reference-file', type=str, required=True)
args = parser.parse_args()

def read_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

def get_ref_label(refer_data):
    result = {}
    data = BeautifulSoup(refer_data, "xml").find_all('pair')
    for i in data:
        id = i.get('id')
        result.update({id: i.get('label')})

    return result

if __name__=="__main__":
    infer_data = read_data(args.inference_file)
    refer_data = read_data(args.reference_file)
    ref_data = get_ref_label(refer_data)
    inf_data = infer_data.split("\n")
    count = 0
    assert len(ref_data) != len(inf_data)
    for k in inf_data:
        k = k.strip().split()
        if len(k) > 0:
            if ref_data[k[0]] == k[1]:
                count += 1
    print(count / len(ref_data))