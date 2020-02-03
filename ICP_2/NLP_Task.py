# Import libraries
import nltk

# The sentences identified to be parsed
sentence_1 = "The dog saw John in the park."
sentence_2 = "The little bear saw the fine fat trout in the rocky brook."

# POS (Part-of-Speech) Tagging
posData_1 = nltk.pos_tag(sentence_1)
print('POS identified in sentence 1 are:')
print(posData_1)

posData_2 = nltk.pos_tag(sentence_2)
print('POS identified in sentence 2 are:')
print(posData_2)

# Tokenization
token_1 = nltk.word_tokenize(sentence_1)
print('Tokens identified within sentence 1 are:')
print(token_1)

token_2 = nltk.word_tokenize(sentence_2)
print('Tokens identified within sentence 2 are:')
print(token_2)

# NER
ner_1 = nlp.ner(sentence_1)
print('Tokens identified within sentence 1 are:')
print(token_1)

token_2 = nltk.word_tokenize(sentence_2)
print('Tokens identified within sentence 2 are:')
print(token_2)

# # Name Entity Recognition (NER) Analysis
# nlp = spacy.load("en_core_web_sm")
# doc_1 = nlp(sentence_1)
# doc_2 = nlp(sentence_2)
# print("Named Entity Recognition for '{}' ".format(sentence_1))
# print("*********************************")
# for ent in doc_1.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)
#
# print("Named Entity Recognition for '{}' ".format(sentence_2))
# print("*********************************")
# for ent in doc_2.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)
#
# # Correlation Recognition Analysis
# neuralcoref.add_to_pipe(nlp)
# print("Co-relation for '{}' ".format(sentence_1))
# print("*********************************")
# print(doc_1._.coref_clusters)
# print("Co-relation for '{}' ".format(sentence_2))
# print("*********************************")
# print(doc_2._.coref_clusters)