from argparse import Namespace

import torch
from transformers import AutoTokenizer, AutoConfig
from model_m import NLLModel


def predict(text, model_path="/Users/young/Library/Mobile Documents/com~apple~CloudDocs/UvA&VU/courses/Web Data Processing Systems/assignment/sota_model/NLL-IE/models/re/re_5.0_epochs.pth"):
    # 加载模型配置和参数
    model_config = "bert-base-cased"  #
    args = Namespace(
        data_dir='./data',
        model_name_or_path=model_config,
        max_seq_length=512,
        batch_size=64,
        learning_rate=6e-5,
        beta1=0.8,
        beta2=0.98,
        eps=1e-6,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        warmup_ratio=0.06,
        num_train_epochs=5.0,
        seed=42,
        num_class=42,
        dropout_prob=0.1,
        project_name="WDPS-RE",
        n_model=2,
        alpha=5.0,
        alpha_warmup_ratio=0.1,

        n_gpu=0
    )

    # 初始化模型
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_class)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = NLLModel(args, config)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs)[0]

    predicted_class = torch.argmax(logits, dim=-1).item()

    return predicted_class


if __name__ == "__main__":
    input_text = "Yang is bob's data, at the same time, Zhi is the professor of University of Amsterdam"
    predicted_label = predict(input_text)

    print(f"Predicted Label: {predicted_label}")
