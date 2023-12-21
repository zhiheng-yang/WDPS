import torch
import random
import numpy as np
import re

from prepro import LABEL_TO_ID
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] + [-1] * (max_len - len(f["labels"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    return output


def predictions_error_modifier(input_list):
    mapping = {1: 2, 3: 4, 5: 6, 7: 8}
    i = 0
    while i < len(input_list):
        current_value = input_list[i]
        replace_value = mapping.get(current_value, None)

        if replace_value is not None:
            j = i + 1
            while j < len(input_list) and input_list[j] == current_value:
                input_list[j] = replace_value
                j += 1

            i = j
        else:
            i += 1

    return input_list


def predict_entities(model, tokenizer, input_text, args):
    entities = []
    current_entity = ""
    current_label = None

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])

    logits = outputs[0]
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    # print("predictions:\n", predictions)
    predictions = predictions_error_modifier(predictions)
    # print("predictions after modifier:\n", predictions)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # print("tokens:\n", tokens[1: len(tokens) - 1])

    for token, pred in zip(tokens, predictions):
        label = ID_TO_LABEL[pred]
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = token
            current_label = label[2:]
        elif label.startswith("I-") and current_label == label[2:]:
            current_entity += token[2:] if token.startswith("##") else (" " + token)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = ""
                current_label = None

    return entities

# TODO: unfinished
def predict_labels(model, tokenizer, input_text, args):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # token_type_ids = inputs.get("token_type_ids", None)
    inputs.pop('token_type_ids', None)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])

    logits = outputs[0]
    predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
    labels = [ID_TO_LABEL[pred] for pred in predictions if pred != LABEL_TO_ID['O']]
    print(labels)
    return labels


def word_tokenize(text):
    words = re.findall(r'\b\w+\b', text)
    return words
