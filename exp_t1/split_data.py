import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

with open('./dataset/c2024/train_labels.json', 'r') as fp:
    train_2024 = json.load(fp)

data = []
for key in train_2024.keys():
    data.append([key, train_2024[key]])

train_2024_df = pd.DataFrame(data, columns=['source', 'target'])

with open('./dataset/c2023/train_labels.json', 'r') as fp:
    train_2023 = json.load(fp)
 
data = []
for key in train_2023.keys():
    data.append([key, train_2023[key]])

train_2023_df = pd.DataFrame(data, columns=['source', 'target'])

train_df, dev_df = train_test_split(train_2023_df, random_state=42, test_size=0.1)

test_df = train_2024_df[[True if x not in train_2023_df['source'].tolist() else False for x in train_2024_df['source']]]

# save train, dev and test data as json
train_data = {}
for index, row in train_df.iterrows():
    train_data[row['source']] = row['target']
    
with open('dataset/train.json', 'w') as fp:
    json.dump(train_data, fp)
    
dev_data = {}
for index, row in dev_df.iterrows():
    dev_data[row['source']] = row['target']
    
with open('dataset/dev.json', 'w') as fp:
    json.dump(dev_data, fp)
    
test_data = {}
for index, row in test_df.iterrows():
    test_data[row['source']] = row['target']
    
with open('dataset/test.json', 'w') as fp:
    json.dump(test_data, fp)