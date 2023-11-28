def getQustion():
    prompt = input("Type your question (for instance: \"The capital of Italy is \") and type ENTER to finish:\n")
    return prompt

# Language model, get an answer from a question
def getAnswer(prompt):
    from ctransformers import AutoModelForCausalLM

    repository="TheBloke/Llama-2-7B-GGUF"
    model_file="llama-2-7b.Q4_K_M.gguf"
    llm = AutoModelForCausalLM.from_pretrained(repository, model_file=model_file, model_type="llama")

    print("Computing the answer (can take some time)...")
    completion = llm(prompt)
    print("COMPLETION: %s" % completion)
    return prompt, completion

# Yes/No/Maybe classifier
def yesNoClassifier(question, context):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


    model = AutoModelForSequenceClassification.from_pretrained("nfliu/roberta-large_boolq")
    tokenizer = AutoTokenizer.from_pretrained("nfliu/roberta-large_boolq")

    # Each example is a (question, context) pair.
    examples = [
        (question, context)
    ]

    encoded_input = tokenizer(examples, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**encoded_input)
        probabilities = torch.softmax(model_output.logits, dim=-1).cpu().tolist()

    probability_no = [round(prob[0], 2) for prob in probabilities]
    probability_yes = [round(prob[1], 2) for prob in probabilities]

    for example, p_no, p_yes in zip(examples, probability_no, probability_yes):
        print(f"Question: {example[0]}")
        print(f"Context: {example[1]}")
        print(f"p(No | question, context): {p_no}")
        print(f"p(Yes | question, context): {p_yes}")
        print()

# fact-checking
def factCheck(claim, evidence):
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

    # Load the tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('Dzeniks/roberta-fact-check')
    model = RobertaForSequenceClassification.from_pretrained('Dzeniks/roberta-fact-check')

    # Define the claim with evidence to classify
    # claim = "Albert Einstein work in the field of computer science"
    # evidence = "Albert Einstein was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time."

    # Tokenize the claim with evidence
    x = tokenizer.encode_plus(claim, evidence, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        prediction = model(**x)

    label = torch.argmax(prediction[0]).item()

    print(f"Label: {label}")

# NER
def ner_bert_large(example):
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from transformers import pipeline

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    # example = "My name is Wolfgang and I live in Berlin"

    ner_results = nlp(example)
    print(ner_results)
    return ner_results


def ner_span_marker(example):
    import spacy
    from spacy import displacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(example)
    displacy.serve(doc, style="ent")

# summarization
def summarization(article):
    from transformers import pipeline

    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    summ = summarizer(article, max_length=1, min_length=1, do_sample=False)
    print(summ)
    return summ[0]['summary_text']


if __name__ == '__main__':
    question = getQustion()
    prompt, completion = getAnswer(question)
    yesNoClassifier(prompt, completion)
    sum_result = summarization(completion)
    # ner_result = ner(sum_result)
    ner_span_marker(sum_result)
