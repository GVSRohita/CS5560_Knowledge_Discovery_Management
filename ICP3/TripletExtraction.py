# Import libraries
import spacy
import textacy

# Read input data in the form of text sentence(s)
text = "Barrack Obama was born in Hawaii"
# text = "'CHICAGO (AP) —Citing high fuel prices, United Airlines said Friday it has increased fares by $6 per round trip on flights to some cities also served by lower-cost carriers. American Airlines, a unit AMR, immediately matched the move, spokesman Tim Wagner said. United, a unit of UAL, said the increase took effect Thursday night and applies to most routes where it competes against discount carriers, such as Chicago to Dallas and Atlanta and Denver to San Francisco, Los Angeles and New York.'"

# Triplet extraction process # 1
nlp = spacy.load('en')
for sentence in text.split("."):
    val = nlp(sentence)
    tuples = textacy.extract.subject_verb_object_triples(val)
    tuples_list = []
    if tuples:
        tuples_to_list = list(tuples)
        tuples_list.append(tuples_to_list)
        print(tuples_list)

# Triplet extraction process # 2
nlp = spacy.load('en')
textExtract = nlp(text)
text_ext = textacy.extract.subject_verb_object_triples(textExtract)
print('Triplets extracted from the input provided are:', text_ext)
