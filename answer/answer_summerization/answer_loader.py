import torch

def load_answer(input, output, tokenizer, model, device):
    combined_input = input + output
    # Encode the combined input
    input_encoding = tokenizer.encode_plus(
        combined_input,
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        truncation=True,
        return_tensors='pt',  # This returns PyTorch tensors
    )
    # Move the tensors to the device
    input_ids = input_encoding['input_ids'].to(device, dtype=torch.long)
    attention_mask = input_encoding['attention_mask'].to(device, dtype=torch.bool)
    # Ensure batch size is at least 1
    if input_ids.size(0) == 0:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
    # Call the model to get predictions for the input
    with torch.no_grad():
        outputs = model(input_ids, attention_mask).squeeze()
        big_val, big_idx = torch.max(outputs, dim=0)
        if big_idx == 1:
            result = "yes"
        elif big_idx == 0:
            result = "no"
        else:
            print("Unexpected prediction value")
    return result
