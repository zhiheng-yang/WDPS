from linking.utilities.entities_format_print import goal_finder, print_answer_entity, print_all_entity
from linking.utilities.linking_loader import linking
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="User provided device_type of 'cuda', but CUDA is not available. Disabling")


# text = "Albert Einstein studied at the Swiss Federal Institute of Technology (ETH Zurich) and later embarked on his remarkable scientific career. He worked as a patent examiner at the Swiss Patent Office in Bern, where he formulated many of his groundbreaking ideas, eventually reshaping our understanding of physics with his theory of relativity."
# text = 'Max Welling is a professor in University of Amsterdam. He won the NIPS 2010 test of time award.'
text = 'Werner Heisenberg, the German physicist and Nobel laureate, ' \
       'studied physics at the University of Munich and later earned his doctorate in theoretical physics at the University of Göttingen. ' \
       'Heisenberg made significant contributions to quantum mechanics and is best known for formulating the Heisenberg Uncertainty Principle. ' \
       'Throughout his career, he held various academic positions, including professorships at the University of Leipzig and the University of Berlin. ' \
       'He also played a key role in the German atomic bomb project during World War II.'

A = "Question: Who is the director of Pulp Fiction? Answer:"
B = "1. everyone's favorite. Answer 2 (for extra credit): Quentin Tarantino, whose directorial debut was Reservoir Dogs."
A = "Question: What is the capital of China? Answer:"
text = """
Computing the answer (can take some time)...
R"
 obviously Beijing.
What is the capital of China?
Beijing is the capital of China.
What is the capitol city of china?
what is the capital of China bejing and why its called that way because it means north of the river
Capital of china is?
The capital of China is Beijing (Peking).
Is Hong Kong the capital of China?
Hong Kong is a special administrative region of China. It is not the capital of China, which is Beijing.
What is the capitol of the country China?
Bejing is the capital city of china
What is the capital city of China?
The capital of China is the Municipality of Beijing (Peking) and the Province of Hebei.
Is Hong Kong a capital of China?
Hong Kong is part of the People's Republic of China, but it is not the capital of China. The capital of China is Beijing.
What is the capital city in China?
Beijing is the capital of China.
How far away from beijing china to Hong kong China?
"""
# TODO: 糾錯，錯別字也能查出來
# print(text)
linking = linking(text)
print(linking)
goal_entity, all_entities = goal_finder(A, B, linking)
print_answer_entity(goal_entity)
print_all_entity(all_entities)
#
# import spacy
#
#
# def extract_entities(text):
#        nlp = spacy.load("en_core_web_sm")
#        doc = nlp(text)
#        entities = [ent.text for ent in doc.ents]
#        return entities
#
#
# def extract_relations(text, entities):
#        nlp = spacy.load("en_core_web_sm")
#        doc = nlp(text)
#
#        relations = []
#        for token in doc:
#               if token.pos_ == "VERB":
#                      subject = next((ent for ent in token.lefts if ent.text in entities), None)
#                      object_ = next((ent for ent in token.rights if ent.text in entities), None)
#
#                      if subject and object_:
#                             relation = {
#                                    "subject": subject.text,
#                                    "relation": token.text,
#                                    "object": object_.text
#                             }
#                             relations.append(relation)
#
#        return relations
#
#
# def generate_knowledge_graph(text):
#        entities = extract_entities(text)
#        relations = extract_relations(text, entities)
#
#        knowledge_graph = {
#               "entities": entities,
#               "relations": relations
#        }
#
#        return knowledge_graph
#
#
# # 示例文本
# input_text = "Max Welling is a professor at the University of Amsterdam."
#
# # 生成知识图谱
# knowledge_graph = generate_knowledge_graph(input_text)
#
# # 输出知识图谱
# print("Entities:", knowledge_graph["entities"])
# print("Relations:")
# for relation in knowledge_graph["relations"]:
#        print(f"{relation['subject']} - {relation['relation']} - {relation['object']}")
