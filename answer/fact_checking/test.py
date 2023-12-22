from fact_checking import *
_evidence = """
    Justine Tanya Bateman (born February 19, 1966) is an American writer, producer, and actress . She is best known for her regular role as Mallory Keaton on the sitcom Family Ties (1982 -- 1989). Until recently, Bateman ran a production and consulting company, SECTION 5 . In the fall of 2012, she started studying computer science at UCLA.
    """
_claim = 'Justine Bateman is a poet.'
print(fact_checking(_claim, _evidence))
