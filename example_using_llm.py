from ctransformers import AutoModelForCausalLM
from ctransformers import AutoConfig
import torch
from answer.answer_summerization.answer_loader import load_answer
from question_detect.question_dectector import detect_question
from transformers import RobertaTokenizer
from answer.modeling_palm import *

from linking.utilities.entities_format_print import goal_finder, print_answer_entity, print_all_entity
from linking.utilities.linking_loader import linking
from transformers import logging
from answer.fact_checking.fact_checking import fact_checking
logging.set_verbosity_error()

boolq_answer_model = torch.load('models/palm_boolq.bin', map_location=torch.device('cpu'))

repository="TheBloke/Llama-2-7B-GGUF"
model_file="llama-2-7b.Q4_K_M.gguf"
llm = AutoModelForCausalLM.from_pretrained(repository, model_file=model_file, model_type="llama")
prompt = input("Type your question (for instance: \"The capital of Italy is \") and type ENTER to finish:\n")
print("Computing the answer (can take some time)...")
completion = llm(prompt)
print("R\"%s\"" % completion)

all_entities = linking(prompt + completion)
if (detect_question(prompt)):
    print(load_answer(prompt, completion, boolq_answer_model))
    print_all_entity(all_entities)
else:
    goal_entity = goal_finder(prompt, completion, all_entities)
    print_answer_entity(goal_entity)
    print_all_entity(all_entities)
print(fact_checking(prompt, completion))



