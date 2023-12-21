import re

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

def print_format_wiki(spans):
    wiki_list = format_wiki(spans)
    for entity in wiki_list:
        print(entity)


