from linking.inference.processor import Refined
from linking.utilities.entities_format_print import print_format_wiki


def linking(text):
    refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                      entity_set="wikipedia")
    spans = refined.process_text(text)
    print_format_wiki(spans)
    return spans
