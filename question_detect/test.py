import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载模型权重文件
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 假设您的模型有两个标签
model.load_state_dict(torch.load('model/question_classification.pt'))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 使用与训练时相同的分词器

# 准备问题文本
question_text = "The capital city of italy is?"

# 分词和编码
inputs = tokenizer(question_text, return_tensors='pt')

# 设置模型为评估模式
model.eval()

# 在没有梯度的情况下进行预测
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

# 可以根据预测的类别来判断问题类型
if predicted_class == 1:
    question_type = "Yes/No"
else:
    question_type = "Open"

print("Predicted Question Type:", question_type)
