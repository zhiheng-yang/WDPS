from linking.inference.processor import Refined
from linking.utilities.entities_format_print import print_format_wiki, format_wiki, fromat_wiki_list, goal_finder


def linking(text, print=False):
    refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                      entity_set="wikipedia")
    spans = refined.process_text(text)
    # formatted_wiki = format_wiki(spans)
    formatted_wiki = fromat_wiki_list(spans)
    if print:
        print_format_wiki(spans)
    return formatted_wiki