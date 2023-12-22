from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# 0 for supports and 1 for refutes
def fact_checking(_claim, _evidence):

    tokenizer = RobertaTokenizer.from_pretrained('Dzeniks/roberta-fact-check')
    model = RobertaForSequenceClassification.from_pretrained('Dzeniks/roberta-fact-check')

    x = tokenizer.encode_plus(_claim, _evidence, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        prediction = model(**x)

    label = torch.argmax(prediction[0]).item()
    return label