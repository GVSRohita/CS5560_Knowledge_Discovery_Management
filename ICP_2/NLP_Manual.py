# Import libraries
from nltk import pos_tag, word_tokenize, ne_chunk
import spacy
import neuralcoref

# Read input data
sentence_1 = "The dog saw John in the park."
sentence_2 = "The little bear saw the fine fat trout in the rocky brook."

# Part-of-speech (POS) tagging
# POS tags of sentence 1
pos_1 = pos_tag(sentence_1)
print('Part-of-speech tags associated with sentence 1:')
print(pos_1)
# POS tags of sentence 2
pos_2 = pos_tag(sentence_2)
print('Part-of-speech tags associated with sentence 2:')
print(pos_2)

# Tokenization
# Tokens of sentence 1
token_1 = word_tokenize(sentence_1)
print('Tokens identified from sentence 1:')
print(token_1)
# Tokens of sentence 2
token_2 = word_tokenize(sentence_2)
print('Tokens identified from sentence 2:')
print(token_2)

# Name entity recognition
# NER of sentence 1
ner_1 = ne_chunk(pos_tag(word_tokenize(sentence_1)))
print('Name entity recognizer of sentence 1:')
print(ner_1)
# NER of sentence 2
ner_2 = ne_chunk(pos_tag(word_tokenize(sentence_2)))
print('Name entity recognizer of sentence 2:')
print(ner_2)

# Co-reference resolution system
nlp = spacy.load("en_core_web_lg")
neuralcoref.add_to_pipe(nlp)
# Co-reference of sentence 1
sent_1 = nlp(sentence_1)
coref_1 = nlp.coref(sent_1)
print('Co-reference identified in sentence 1:')
print(coref_1)
# Co-reference of sentence 2
sent_2 = nlp(sentence_2)
coref_2 = nlp.coref(sent_2)
print('Co-reference identified in sentence 2:')
print(coref_2)

