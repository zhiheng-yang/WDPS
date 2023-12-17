
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
    model_path = "/Users/young/Library/Mobile Documents/com~apple~CloudDocs/UvA&VU/courses/Web Data Processing Systems/assignment/sota_model/NLL-IE/ner/saved_models/ner_40_epoch.pth"
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
    # input_text = "Jacopo is a person. He is a professor at the University of Amsterdam. " \
                 # "He is also a researcher at the Vrije Universiteit Amsterdam."
    input_text = 'England won the FIFA World Cup in 1966.'
    input_text = 'Max Welling is a football player in University of Amsterdam. He lead the Netherlands won the NIPS 2010 test of time award.'
    input_text = "Yes, Managua is the capital city of Nicaragua. It is located in the southwestern part of the country and is home to many important government buildings and institutes, including the President's office and the National Assembly. The city has a popula!on of over one million people and is known for its vibrant cultural scene, historic landmarks, and beau!ful natural surroundings."

    entities = predict_entities(model, tokenizer, input_text, args)

    print("Entities:", entities)


if __name__ == "__main__":
    main()