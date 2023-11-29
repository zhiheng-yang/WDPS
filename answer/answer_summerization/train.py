import gzip
import random

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from palm_pytorch import PaLM
from autoclassifier_wapper import AutoClassifierWrapper
import json
from transformers import AutoTokenizer

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 1024

LABEL_TO_ID = {True: 1, False: 2}

# helpers


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# instantiate GPT-like decoder model

model = PaLM(num_tokens=256, dim=512, depth=8)

model = AutoClassifierWrapper(model, max_seq_len=SEQ_LEN)
# model.cuda()

# prepare enwik8 data

class JsonDataset(Dataset):
    def __init__(self, file_path, seq_len, tokenizer):
        super().__init__()
        # Read JSONL file and store data
        self.data = []
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        with open(file_path, 'r') as file:
            for line in file:
                item = json.loads(line)
                self.data.append(item)

    def __getitem__(self, index):
        # Extract and return a sample
        sample = self.data[index]
        question = sample['question']
        answer = sample['answer']
        passage = sample['passage']

        # Tokenize text
        tokenized_question = self.tokenizer.tokenize(question)[:self.seq_len - 2]
        tokenized_answer = [LABEL_TO_ID[answer]]
        tokenized_passage = self.tokenizer.tokenize(passage)[:self.seq_len - 2]

        # Convert the tokens to their corresponding token IDs
        input_question = self.tokenizer.convert_tokens_to_ids(tokenized_question)
        # Add special tokens to the input sequence
        input_question = self.tokenizer.build_inputs_with_special_tokens(input_question)

        input_passage = self.tokenizer.convert_tokens_to_ids(tokenized_passage)
        input_passage = self.tokenizer.build_inputs_with_special_tokens(input_passage)

        result = {'question': input_question, 'passage': input_passage, 'answer': tokenized_answer}
        return result

    def __len__(self):
        return len(self.data)
    

def collate_fn(batches):
    # Find the maximum length of question in the batch
    max_len_q = max([len(batch["question"]) for batch in batches])
    # Find the maximum length of passage in the batch
    max_len_p = max([len(batch["passage"]) for batch in batches])

    # Pad input_ids sequences to the maximum length in the batch
    question = [batch["question"] + [0] * (max_len_q - len(batch["question"])) for batch in batches]
    
    # Pad input_ids sequences to the maximum length in the batch
    passage = [batch["passage"] + [0] * (max_len_p - len(batch["passage"])) for batch in batches]

    answer = [batch["answer"] for batch in batches]

    answer = torch.LongTensor(answer)
    passage = torch.LongTensor(passage)
    question = torch.LongTensor(question)
   
    output = {
        'question': question,
        'passage': passage,
        'answer': answer,
    }
    return output

data_train = "../../datasets/boolq/train.jsonl"
data_val = "../../datasets/boolq/dev.jsonl"
tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/roberta-base-boolq")
train_dataset = JsonDataset(data_train, SEQ_LEN, tokenizer)
val_dataset = JsonDataset(data_val, SEQ_LEN, tokenizer)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=True))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=True))
# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    print(f"start training: {i}")
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        next_item = next(train_loader)
        print(next_item['answer'])
        print(next_item['answer'])
        print(next_item['answer'])
        loss = model(question=next_item['question'], passage=next_item['passage'], answer=next_item['answer'])
        loss.backward()

    print(f"training loss: {loss.item()}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            data_item = next(val_loader)
            print(data_item.passage)
            loss = model(question=data_item.question, passage=data_item.passage, answer=data_item.answer)
            print(f"validation loss: {loss.item()}")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(output_str)