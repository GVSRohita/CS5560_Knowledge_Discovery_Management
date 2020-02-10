# importing the library
from nltk.corpus import wordnet

# Get synsets for the words
# Word: good
good = wordnet.synsets("good")
print('Synsets for the word Good:')
print(good)
# Word: look
look = wordnet.synsets("paying")
print('Synsets for the word Look:')
print(look)

# Hyponyms
print('Hyponyms of the word GOOD:')
print(wordnet.synset('good.n.01').hyponyms())
print(wordnet.synset('good.n.02').hyponyms())
print(wordnet.synset('good.n.03').hyponyms())

# Hypernyms
print('Hypernyms of the word GOOD:')
print(wordnet.synset('good.n.01').hypernyms())
print(wordnet.synset('good.n.02').hypernyms())
print(wordnet.synset('good.n.03').hypernyms())

# Meronym
print('Meronyms of the word GOOD:')
print(wordnet.synset('kitchen.n.01').part_meronyms())
print(wordnet.synset('good.n.02').part_meronyms())
print(wordnet.synset('good.n.03').part_meronyms())

# Holonym
print('Holonym of the word GOOD:')
print(wordnet.synset('kitchen.n.01').part_holonyms())
print(wordnet.synset('good.n.02').part_holonyms())
print(wordnet.synset('good.n.03').part_holonyms())

# Entailment
print('Entailment of the word LOOK:')
print(wordnet.synset('snore.v.01').entailments())
print(wordnet.synset('give.v.05').entailments())
print(wordnet.synset('yield.v.10').entailments())