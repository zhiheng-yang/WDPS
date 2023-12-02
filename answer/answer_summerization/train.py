import gzip
import random
from einops import rearrange
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm, trange
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from palm_pytorch import PaLM
from autoclassifier_wapper import AutoClassifierWrapper
import json
from transformers import AutoTokenizer
from dataset_utils import convert_examples_to_features, processors

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

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

# model.cuda()

processor = processors["json"]
data_train = "../../datasets/boolq/train.jsonl"
data_val = "../../datasets/boolq/dev.jsonl"
label_map = processor.get_labels()
tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/roberta-base-boolq")
train_examples = processor.get_train_examples(data_train)
val_examples = processor.get_dev_examples(data_val)

train_features = convert_examples_to_features(train_examples, label_map, SEQ_LEN, tokenizer)
val_features = convert_examples_to_features(val_examples, label_map, SEQ_LEN, tokenizer)

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_attn_mask= torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

train_data = TensorDataset(all_input_ids, all_attn_mask, all_label_ids)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

# # optimizer
device = torch.device("cpu")

model = PaLM(dim=SEQ_LEN, num_tokens=tokenizer.vocab_size, depth=8, num_labels=2)

model = AutoClassifierWrapper(model, max_seq_len=SEQ_LEN)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
num_train_optimization_steps = len(train_dataloader)
print(num_train_optimization_steps)
# training

for i in trange(int(100), desc="Epoch"):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attn_mask, label_ids = batch
        model_out = model(x = input_ids)
        # print(model_out.shape)
        # print(label_ids.view(-1))
        loss = criterion(model_out.view(-1, 2), label_ids.view(-1))
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:-1]
        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        sample = model.generate(inp[None, ...], GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print(output_str)