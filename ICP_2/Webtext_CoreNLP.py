# Importing requests library to send HTTP/1.1 requests
import requests

# Variable assigned to url of website from which data is to be scrapped
wikipage = requests.get('https://en.wikipedia.org/wiki/Text_mining').text

# Parsing data using BeautifulSoup function
from bs4 import BeautifulSoup
soup = BeautifulSoup(wikipage, 'html')

# Identifying text data accessed from web page
textData = soup.get_text()

# Import appropriate libraries
import neuralcoref
import spacy
from stanfordcorenlp import StanfordCoreNLP

# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port,timeout=30000)

# Part-of-speech (POS) tagging
print('POS tag identified from text obtained from web page:')
print(nlp.pos_tag(textData))

# Tokenize
print('Tokens identified from text obtained from web page')
print(nlp.word_tokenize(textData))

# Name entity recognizer (NER)
print('Name entity recognizer identified from text obtained from web page')
print('NER：', nlp.ner(textData))

# Parser
print('Parser assocaited with text obtained from web page：')
print(nlp.parse(textData))
print(nlp.dependency_parse(textData))

# Co-reference resolution system
nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)
# Co-reference of sentence 1
txt = nlp(textData)
print('Co-reference identified in text obtained from web page:')
print(txt._.coref_clusters)

# Close Stanford Parser
nlp.close()


