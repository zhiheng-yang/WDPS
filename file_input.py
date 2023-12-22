import sys
from ctransformers import AutoModelForCausalLM
from ctransformers import AutoConfig
import torch
from answer.answer_summerization.answer_loader import load_answer
from question_detect.question_dectector import detect_question
import argparse
from answer.modeling_palm import *

from linking.utilities.entities_format_print import goal_finder, print_answer_entity, print_all_entity
from linking.utilities.linking_loader import linking
from transformers import logging
from answer.fact_checking.fact_checking import fact_checking
logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
boolq_answer_model = torch.load('/app/models/palm_boolq.bin', map_location=torch.device(device))

repository="TheBloke/Llama-2-7B-GGUF"
model_file="llama-2-7b.Q4_K_M.gguf"
llm = AutoModelForCausalLM.from_pretrained(repository, model_file=model_file, model_type="llama")

def process_line(line_counter, line):
    line = line.strip()  # Remove leading/trailing whitespace
    # Print the input line with its index
    print(f"Question-{line_counter:03}\t{line}")
    completion = llm(line)
    print(f"Question-{line_counter:03}\tR\"{completion}\"")

    all_entities = linking(line + completion)
    if (detect_question(line)):
        yes_no = load_answer(line, completion, boolq_answer_model)
        print(f"Question-{line_counter:03}\t{yes_no}")

    else:
        goal_entity = goal_finder(line, completion, all_entities)
        print(f"Question-{line_counter:03}\tE\"{goal_entity[0]}\"<TAB>\"{goal_entity[1]}\"")
    for entity in all_entities:
        print(f"Question-{line_counter:03}\tE\"{entity[0]}\"<TAB>\"{entity[1]}\"")
    fact = fact_checking(line, completion)
    print(f"Question-{line_counter:03}\t{fact}")

def main(input_file):
    try:
        with open(input_file, 'r') as file:
            for line_counter, line in enumerate(file, start=1):
                process_line(line_counter, line)
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)

    main(sys.argv[1])
