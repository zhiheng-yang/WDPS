import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
logging.basicConfig(level=logging.ERROR)

# load model weight model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2) 
model.load_state_dict(torch.load('/app/models/answer_classification.pt', map_location=torch.device(device)))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_answer(input, output, palm_model):
    inputs = tokenizer(input, output, max_length=256, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    if predicted_class == 1:
        result = "A\"yes\""
    elif predicted_class == 0:
        result = "A\"no\""
    else:
        print("Unexpected prediction value")
    return result


# import torch
# from torch import cuda
# from transformers import RobertaTokenizer
# import warnings

# warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp.autocast_mode")

# MAX_LEN = 256
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)


# def load_answer(input, output, model):
#     combined_input = input + output
#     # Encode the combined input
#     input_encoding = tokenizer.encode_plus(
#         combined_input,
#         add_special_tokens=True,
#         max_length=256,
#         padding=True,
#         truncation=True,
#         return_tensors='pt',  # This returns PyTorch tensors
#     )
#     device = 'cuda' if cuda.is_available() else 'cpu'
#     # Move the tensors to the device
#     input_ids = input_encoding['input_ids'].to(device, dtype=torch.long)
#     attention_mask = input_encoding['attention_mask'].to(device, dtype=torch.bool)
#     # Ensure batch size is at least 1
#     if input_ids.size(0) == 0:
#         input_ids = input_ids.unsqueeze(0)
#         attention_mask = attention_mask.unsqueeze(0)
#     # Call the model to get predictions for the input
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask).squeeze()
#         big_val, big_idx = torch.max(outputs, dim=0)
#         if big_idx == 1:
#             result = "A\"yes\""
#         elif big_idx == 0:
#             result = "A\"no\""
#         else:
#             print("Unexpected prediction value")
#     return result
