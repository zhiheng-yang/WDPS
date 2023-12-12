import logging
import torch
import pandas as pd
import json
from transformers import RobertaModel, RobertaTokenizer
from boolq_data_loader import boolq_data_loader
import sys
sys.path.append('..')

from modeling_palm import AutoClassifierWrapper
from classifier_util import device, loss_function, train, valid

logging.basicConfig(level=logging.ERROR)
# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
# EPOCHS = 1
LEARNING_RATE = 1e-05
EPOCHS = 12

TRAINING_PATH = "../../datasets/boolq/train.jsonl"
TEST_PATH = "../../datasets/boolq/dev.jsonl"

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = boolq_data_loader(TRAINING_PATH, tokenizer, MAX_LEN, train_params)
testing_loader = boolq_data_loader(TEST_PATH, tokenizer, MAX_LEN, test_params)
model = AutoClassifierWrapper(dim = MAX_LEN,
                              vocab_size= tokenizer.vocab_size,
                              depth=2,
                              dim_head=64,
                              heads = 8,
                              ff_mult=4,
                              max_seq_len=MAX_LEN)
model.to(device)

# Creating the loss function and optimizer
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    train(epoch, model, training_loader, optimizer)
acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)
output_model_file = 'palm_bool.bin'
output_vocab_file = './'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')