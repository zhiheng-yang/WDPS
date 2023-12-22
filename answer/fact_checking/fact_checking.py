from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
tokenizer = RobertaTokenizer.from_pretrained('Dzeniks/roberta-fact-check')
model = RobertaForSequenceClassification.from_pretrained('Dzeniks/roberta-fact-check')
label_to_message = {0: "C\"correct\"", 1: "C\"incorrect\""}

# 0 for supports and 1 for refutes
def fact_checking(_claim, _evidence, local=True):

    model_name = "JLei/climate_fever_roberta-base-fact-checking" if local else 'Dzeniks/roberta-fact-check'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    x = tokenizer.encode_plus(_claim, _evidence, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        prediction = model(**x)

    label = torch.argmax(prediction[0]).item()
    return label_to_message[label]