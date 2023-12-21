from linking.utilities.entities_format_print import goal_finder, print_answer_entity, print_all_entity
from linking.utilities.linking_loader import linking

# text = "Albert Einstein studied at the Swiss Federal Institute of Technology (ETH Zurich) and later embarked on his remarkable scientific career. He worked as a patent examiner at the Swiss Patent Office in Bern, where he formulated many of his groundbreaking ideas, eventually reshaping our understanding of physics with his theory of relativity."
# text = 'Max Welling is a professor in University of Amsterdam. He won the NIPS 2010 test of time award.'
text = 'Werner Heisenberg, the German physicist and Nobel laureate, ' \
       'studied physics at the University of Munich and later earned his doctorate in theoretical physics at the University of Göttingen. ' \
       'Heisenberg made significant contributions to quantum mechanics and is best known for formulating the Heisenberg Uncertainty Principle. ' \
       'Throughout his career, he held various academic positions, including professorships at the University of Leipzig and the University of Berlin. ' \
       'He also played a key role in the German atomic bomb project during World War II.'

A = "Question: Who is the director of Pulp Fiction? Answer:"
B = "1. everyone's favorite. Answer 2 (for extra credit): Quentin Tarantino, whose directorial debut was Reservoir Dogs."
text = A + B
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
