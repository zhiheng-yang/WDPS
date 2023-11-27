
import argparse

from extraction.NER.ner_util import predictions_error_modifier, predict_entities
# from prepro import LABEL_TO_ID
import torch
from transformers import AutoTokenizer
from model_m import NLLModel


# TODO: haven't finish, but can be used for test
def load_model(model_path, args):
    model = NLLModel(args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def main():
    model_path = "../../models/ner/ner.pth"
    args = argparse.Namespace(
        data_dir="../../datasets/CoNLL",
        model_name_or_path="bert-base-cased", # or bert-large-cased
        max_seq_length=512,
        batch_size=64,
        learning_rate=1e-5,
        gradient_accumulation_steps=1,
        eps=1e-6,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        dropout_prob=0.1,
        num_train_epochs=1.0,
        seed=42,
        num_class=9,
        project_name="WDPS-NER",
        n_model=1,
        alpha=50.0,
        alpha_warmup_ratio=0.1
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model = load_model(model_path, args)

    # ['J', '##acopo', 'University of Amsterdam', 'Vrije Universiteit Amsterdam']
    input_text = "Jacopo is a person. He is a professor at the University of Amsterdam. " \
                 "He is also a researcher at the Vrije Universiteit Amsterdam."
    # input_text = 'Jocapo is a professor in Vrije University Amsterdam.'

    entities = predict_entities(model, tokenizer, input_text, args)

    print("Entities:", entities)


if __name__ == "__main__":
    main()