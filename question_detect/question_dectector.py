import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import logging
logging.set_verbosity_error()

# load model weight model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) 
model.load_state_dict(torch.load('question_detect/question_classification.pt', map_location=torch.device(device)))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def detect_question(question):
    inputs = tokenizer(question, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class == 1