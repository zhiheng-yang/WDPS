from ctransformers import AutoModelForCausalLM
from ctransformers import AutoConfig
import torch
from answer.answer_summerization.answer_loader import load_answer
from answer.question_detection.question_dectector import detect_question
from transformers import RobertaTokenizer
from answer.modeling_palm import *
question_detection_model = torch.load('models/palm_question.bin', map_location=torch.device('cpu'))
boolq_answer_model = torch.load('models/palm_boolq.bin', map_location=torch.device('cpu'))
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)

repository="TheBloke/Llama-2-7B-GGUF"
model_file="llama-2-7b.Q4_K_M.gguf"
llm = AutoModelForCausalLM.from_pretrained(repository, model_file=model_file, model_type="llama")
prompt = input("Type your question (for instance: \"The capital of Italy is \") and type ENTER to finish:\n")
print("Computing the answer (can take some time)...")
llm.config.context_length = 10
completion = llm(prompt)
print("COMPLETION: %s" % completion)

if (detect_question(prompt, tokenizer, question_detection_model, 128)):
    print(load_answer(prompt, completion, tokenizer, boolq_answer_model, 512))
