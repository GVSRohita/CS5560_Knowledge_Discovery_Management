from stanfordcorenlp import StanfordCoreNLP

# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port,timeout=30000)

# The sentence you want to parse
sentence_1 = "The dog saw John in the park"
sentence_2 = "The little bear saw the fine fat trout in the rocky brook"

# POS
# POS (Part-of-Speech) Tagging
posData_1 = nlp.pos_tag(sentence_1)
print('POS identified in sentence 1 are:')
print(posData_1)

posData_2 = nlp.pos_tag(sentence_2)
print('POS identified in sentence 2 are:')
print(posData_2)

# Tokenize
token_1 = nlp.word_tokenize(sentence_1)
print('Tokens identified within sentence 1 are:')
print(token_1)

token_2 = nlp.word_tokenize(sentence_2)
print('Tokens identified within sentence 2 are:')
print(token_2)

# NER
ner_1 = nlp.ner(sentence_1)
print('Tokens identified within sentence 1 are:')
print(ner_1)

ner_2 = nlp.ner(sentence_2)
print('Tokens identified within sentence 2 are:')
print(ner_2)

# Correlation
corr_1 = nlp.coref(sentence_1)
print('Tokens identified within sentence 1 are:')
print(corr_1)

corr_2 = nlp.coref(sentence_2)
print('Tokens identified within sentence 2 are:')
print(corr_2)

# Close Stanford Parser
nlp.close()