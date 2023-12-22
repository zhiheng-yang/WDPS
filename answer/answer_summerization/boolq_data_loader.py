import json
import pandas as pd
import sys
sys.path.append('..')
from classifier_data import ClassifierData
from torch.utils.data import Dataset, DataLoader

def boolq_data_loader(file_path, tokenizer, max_len, load_params):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame(data)
    df['text'] = df['question'] + '?'  + df['passage']
    df['label'] = df['answer'].replace({False: 0, True: 1})
    df_data = df[['text', 'label']].copy()
    data_set = ClassifierData(df_data, tokenizer, max_len)
    data_loader = DataLoader(data_set, **load_params)
    return data_loader

