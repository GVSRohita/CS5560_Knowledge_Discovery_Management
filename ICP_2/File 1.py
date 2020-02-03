from stanfordcorenlp import StanfordCoreNLP

# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port,timeout=30000)

# The sentence you want to parse
sentence = 'Citing high fuel prices, United Airlines said Friday it has increased fares by $6 per round trip on flights to some cities also served by lower-cost carriers. American Airlines, a unit AMR, immediately matched the move, spokesman Tim Wagner said. United, a unit of UAL, said the increase took effect Thursday night and applies to most routes where it competes against discount carriers, such as Chicago to Dallas and Atlanta and Denver to San Francisco, Los Angeles and New York.'

# POS
print('POS：', nlp.pos_tag(sentence))

# Tokenize
print('Tokenize：', nlp.word_tokenize(sentence))

# NER
print('NER：', nlp.ner(sentence))

# Parser
print('Parser：')
print(nlp.parse(sentence))
print(nlp.dependency_parse(sentence))

# Correlation
print()

# Close Stanford Parser
nlp.close()