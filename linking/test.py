from linking.utilities.linking_loader import linking

text = "Yes, Managua is the capital city of Nicaragua. " \
       "It is located in the southwestern part of the country and " \
       "is home to many important government buildings and institutes, " \
       "including the Nicaraguan President's Office and the National Assembly. " \
       "The city has a population of over one million people and is known for its vibrant cultural scene, " \
       "historic landmarks, and beautiful natural surroundings."
# text = 'Max Welling is a professor in University of Amsterdam. He won the NIPS 2010 test of time award.'

linking = linking(text)