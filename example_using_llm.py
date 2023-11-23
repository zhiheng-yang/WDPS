from ctransformers import AutoModelForCausalLM

repository="TheBloke/Llama-2-7B-GGUF"
model_file="llama-2-7b.Q4_K_M.gguf"
llm = AutoModelForCausalLM.from_pretrained(repository, model_file=model_file, model_type="llama")

prompt = input("Type your question (for instance: \"The capital of Italy is \") and type ENTER to finish:\n")
print("Computing the answer (can take some time)...")
completion = llm(prompt)
print("COMPLETION: %s" % completion)
