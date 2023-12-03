import gzip
import os
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
data_train = "/app/datasets/boolq/train.jsonl"
data_val = "/app/datasets/boolq/dev.jsonl"
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

val_input_ids = torch.tensor([f.input_ids for f in val_features], dtype=torch.long)
val_attn_mask= torch.tensor([f.input_mask for f in val_features], dtype=torch.long)
val_label_ids = torch.tensor([f.label_id for f in val_features], dtype=torch.long)
val_data = TensorDataset(val_input_ids, val_attn_mask, val_label_ids)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

# # optimizer
device = torch.device("cpu")

model = PaLM(dim=SEQ_LEN, num_tokens=tokenizer.vocab_size, depth=8, num_labels=2)

model = AutoClassifierWrapper(model, max_seq_len=SEQ_LEN)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
num_train_optimization_steps = len(train_dataloader)
print(num_train_optimization_steps)
# training

print("starting")
for i in trange(int(50), desc="Epoch"):
    print(f"{i=}")
    epoch_loss = 0
    epoch_correct = 0
    epoch_count = 0
    preds = []
    out_label_ids = None
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attn_mask, label_ids = batch
        model_out = model(x = input_ids)
        # print(model_out.shape)
        # print(label_ids.view(-1))
        loss = criterion(model_out.view(-1, 2), label_ids.view(-1))

        correct = model_out.argmax(axis=1) == label_ids
        acc = correct.sum().item() / correct.size(0)
        epoch_correct += correct.sum().item()
        epoch_count += correct.size(0)

        epoch_loss += loss.item()
        if len(preds) == 0:
            preds.append(model_out.detach().cpu().numpy())
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds[0] = np.append(
                preds[0], model_out.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    result = {'acc': (preds == out_label_ids).mean()}
    print(f"epoch accuracy: {result['acc']}")
    model_dir = "/app/models/as"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"as_{i}_epochs.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")
    with torch.no_grad():
        test_epoch_loss = 0
        test_epoch_correct = 0
        test_epoch_count = 0
        preds = []
        out_label_ids = None

        model.eval()
        for input_ids, attn_mask, label_ids in tqdm(val_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            label_ids = label_ids.to(device)
            model_out = model(x = input_ids)
            test_loss = criterion(model_out.view(-1, 2), label_ids.view(-1))
            if len(preds) == 0:
                preds.append(model_out.detach().cpu().numpy())
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                preds[0] = np.append(
                    preds[0], model_out.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, label_ids.detach().cpu().numpy(), axis=0)
            correct = model_out.argmax(axis=1) == label_ids
            test_epoch_correct += correct.sum().item()
            test_epoch_count += correct.size(0)
            test_epoch_loss += loss.item()
        preds = preds[0]
        preds = np.argmax(preds, axis=1)
        result = {'acc': (preds == out_label_ids).mean()}
        print(f"{epoch_loss=}")
        print(f"epoch accuracy: {result['acc']}")
        print(f"{test_epoch_loss=}")
        print(f"test epoch accuracy: {test_epoch_correct / test_epoch_count}")