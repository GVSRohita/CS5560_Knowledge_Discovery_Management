# Importing requests library to send HTTP/1.1 requests
import requests

# Variable assigned to url of website from which data is to be scrapped
wikipage = requests.get('https://en.wikipedia.org/wiki/Text_mining').text

# Parsing data using BeautifulSoup function
from bs4 import BeautifulSoup
soup = BeautifulSoup(wikipage, 'html')

# Finding all the links within the page containing a tag
sentence_1 = soup.get_text()

from nltk import pos_tag, word_tokenize, ne_chunk
import spacy
import neuralcoref

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


