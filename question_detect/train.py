from datasets import load_dataset
from datasets import concatenate_datasets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from question_data_loader import question_data_loader
# Load the trec and val dataset
combined_dataset = question_data_loader('train', 'train')
val_combined_dataset = question_data_loader('test', 'validation')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# tokenizer the dataset
combined_dataset = combined_dataset.map(lambda e: tokenizer(e['question'], truncation=True, padding='max_length', max_length=128), batched=True)
val_combined_dataset = val_combined_dataset.map(lambda e: tokenizer(e['question'], truncation=True, padding='max_length', max_length=128), batched=True)

combined_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
val_combined_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

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
    logging_steps=len(combined_dataset) // 8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset,
    eval_dataset=val_combined_dataset,
    compute_metrics = compute_metrics
)

trainer.train()
# save
torch.save(model.state_dict(), 'question_classification.pt')
# evaluate
eval_result = trainer.evaluate()
print(eval_result)