from datasets import load_dataset, concatenate_datasets

def question_data_loader(trec_split, boolq_split):
    train_trec_dataset = load_dataset("trec", split=trec_split)
    train_boolq_dataset = load_dataset("boolq", split=boolq_split)
    # Add a question mark to the end of boolq question in the train dataset
    train_boolq_dataset = train_boolq_dataset.map(lambda example: {"question": [q + " ?" for q in example["question"]]}, batched=True)
    train_trec_dataset = train_trec_dataset.map(lambda example: {'question': example['text'], 'label': 0})
    # Ensure both datasets only have 'question' and 'label' columns
    train_boolq_dataset = train_boolq_dataset.remove_columns([col for col in train_boolq_dataset.column_names if col not in ['question', 'label']])
    train_trec_dataset = train_trec_dataset.remove_columns([col for col in train_trec_dataset.column_names if col not in ['question', 'label']])
    # Concatenate the datasets
    combined_dataset = concatenate_datasets([train_boolq_dataset, train_trec_dataset])

    # Shuffle the combined dataset
    combined_dataset = combined_dataset.shuffle()
    return combined_dataset

