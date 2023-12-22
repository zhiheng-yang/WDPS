import re
from linking.data_types.base_types import Entity, Span

def print_entities(spans):
    entities_with_links = [span.text+'<TAB>'+span.predicted_entity.wikidata_entity_id for span in spans if span.predicted_entity.wikidata_entity_id is not None]
    for entity in entities_with_links:
        print(entity)




def format_wikipedia_title(title):
    formatted_title = re.sub(r'\s+', '_', title)
    return formatted_title

def return_wikipedia_url(title, formatted=True):
    formatted_title = format_wikipedia_title(title)
    if formatted:
        return 'https://en.wikipedia.org/wiki/'+formatted_title
    else:
        return 'https://en.wikipedia.org/wiki/'+title

def format_wiki(spans):
    # entities_with_links = [span.text+'<TAB>'+span.predicted_entity.wikipedia_entity_title for span in spans
                           # if (span.candidate_entities is not None or span.candidate_entities != [])]
    wiki_list = []
    for span in spans:
        # print(span)
        if span.predicted_entity is not None:
            if span.predicted_entity.wikipedia_entity_title is not None:
                wiki_list.append(span.text+'<TAB>'+return_wikipedia_url(span.predicted_entity.wikipedia_entity_title))
            else:
                # wiki_list.append(span.text+'<TAB>'+span.__repr__())
                wiki_list.append(span.text+'<TAB>'+'Entity not linked to a knowledge base')
    return wiki_list

def deduplicate_spans(span_list: list[Span]) -> list[Span]:
    seen_links = set()
    unique_spans = []

    for span in span_list:
        link = span.text

        if link not in seen_links:
            seen_links.add(link)
            unique_spans.append(span)

    return unique_spans

# also for duplicate spans
def fromat_wiki_list(spans):
    # deduplicated_spans = deduplicate_spans(spans)
    wiki_list = []
    for span in spans:
        # print(span)
        if span.predicted_entity is not None:
            if span.predicted_entity.wikipedia_entity_title is not None:
                if not (span.text in [entity[0] for entity in wiki_list] \
                        and return_wikipedia_url(span.predicted_entity.wikipedia_entity_title) in [entity[1] for entity in wiki_list]):
                    wiki_list.append([span.text, return_wikipedia_url(span.predicted_entity.wikipedia_entity_title)])
            else:
                if span.text not in [entity[0] for entity in wiki_list]:
                    # wiki_list.append(span.text+'<TAB>'+span.__repr__())
                    wiki_list.append([span.text, 'Entity not linked to a knowledge base'])
    return wiki_list

def print_format_wiki(spans):
    wiki_list = format_wiki(spans)
    for entity in wiki_list:
        print(entity)


def goal_finder(input_a: str, output_b: str, entities: list):
    entities_a = [entity for entity in entities if (entity[0] in input_a)]
    entities_b = [entity for entity in entities if entity not in entities_a]
    goal = entities_b[0]
    return goal, entities

def print_answer_entity(entity):
    print("A\""+entity[0]+'\"<TAB>\"'+entity[1]+"\"")

def print_all_entity(entities):
    for entity in entities:
        print("E\""+entity[0]+'\"<TAB>\"'+entity[1]+"\"")
