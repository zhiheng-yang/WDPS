import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch


data = pd.read_csv('dataset/train.csv')

print(data.head())
data.dropna(inplace=True)

# lowercase
data['question'] = data['question'].str.lower()

#split the dataset
train, test = train_test_split(data, test_size=0.2)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# compute the metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
def encode(df):
    return tokenizer.batch_encode_plus(
        df['question'].tolist(),
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

train_encodings = encode(train)
test_encodings = encode(test)


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = QuestionDataset(train_encodings, train['label'].tolist())
test_dataset = QuestionDataset(test_encodings, test['label'].tolist())

# start training
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 假设有两个标签

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1,
    evaluation_strategy='epoch',
    logging_steps=len(train_dataset) // 8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics = compute_metrics
)

trainer.train()
# save
torch.save(model.state_dict(), 'model/question_classification.pt')
# evaluate
eval_result = trainer.evaluate()
print(eval_result)