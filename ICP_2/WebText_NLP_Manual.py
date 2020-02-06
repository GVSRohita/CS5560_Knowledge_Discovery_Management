# Import libraries
from nltk import pos_tag, word_tokenize, ne_chunk
import spacy
import neuralcoref

# Read input data
sentence_1 = "http://textfiles.com/adventure/aencounter.txt"

# Part-of-speech (POS) tagging
# POS tags of sentence 1
pos_1 = pos_tag(sentence_1)
print('Part-of-speech tags associated with sentence 1:')
print(pos_1)

# Tokenization
# Tokens of sentence 1
token_1 = word_tokenize(sentence_1)
print('Tokens identified from sentence 1:')
print(token_1)

# Name entity recognition
# NER of sentence 1
ner_1 = ne_chunk(pos_tag(word_tokenize(sentence_1)))
print('Name entity recognizer of sentence 1:')
print(ner_1)

# Co-reference resolution system
nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)
# Co-reference of sentence 1
sent_1 = nlp(sentence_1)
print('Co-reference identified in sentence 1:')
print(sent_1._.coref_clusters)


