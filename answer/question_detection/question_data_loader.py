import json
import pandas as pd
import sys
sys.path.append('..')
from classifier_data import ClassifierData
from torch.utils.data import Dataset, DataLoader

def question_data_loader(file_path, tokenizer, max_len, load_params):
    df = pd.read_csv(file_path)
    df = df.rename(columns={'question': 'text'})
    df = df[['text', 'label']]
    data_set = ClassifierData(df, tokenizer, max_len)
    data_loader = DataLoader(data_set, **load_params)
    return data_loader

