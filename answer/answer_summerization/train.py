import torch
import os
from transformers import RobertaModel, RobertaTokenizer
from boolq_data_loader import boolq_data_loader
import sys
import argparse
sys.path.append('..')

from modeling_palm import AutoClassifierWrapper
from classifier_util import device, loss_function, train, valid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../datasets/boolq", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--depth", default=6, type=int)
    parser.add_argument("--eps", default=12, type=float)
    parser.add_argument("--dim_head", default=64, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--ff_multi", default=4, type=int)
    parser.add_argument("--num_train_epochs", default=30, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_class", type=int, default=9)

    parser.add_argument("--project_name", type=str, default="NLL-IE-NER")
    parser.add_argument("--n_model", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    args = parser.parse_args()
    
    TRAINING_PATH = os.path.join(args.data_dir, "train.jsonl")
    TEST_PATH = os.path.join(args.data_dir, "dev.jsonl")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path, truncation=True, do_lower_case=True)

    train_params = {'batch_size': args.batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': args.batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    training_loader = boolq_data_loader(TRAINING_PATH, tokenizer, args.max_seq_length, train_params)
    testing_loader = boolq_data_loader(TEST_PATH, tokenizer, args.max_seq_length, test_params)
    model = AutoClassifierWrapper(dim = args.max_seq_length,
                              vocab_size= tokenizer.vocab_size,
                              depth=args.depth,
                              dim_head=args.dim_head,
                              heads = args.heads,
                              ff_mult=args.ff_mult,
                              max_seq_len=args.max_seq_length)
    model.to(device)

    # Creating the loss function and optimizer
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_train_epochs):
        train(epoch, model, training_loader, optimizer)
    acc = valid(model, testing_loader)
    print("Accuracy on test data = %0.2f%%" % acc)
    output_model_file = 'palm_bool.bin'
    output_vocab_file = './'

    model_to_save = model
    torch.save(model_to_save, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)

    print('All files saved')

if __name__ == "__main__":
     main()