from __future__ import print_function
import spacy
import textacy
import os

new_list = []
for root, dirs, files in os.walk("Abstracts Input/*"):
    print(files)
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                new_list.append(text)

# print(new_list)
clean_data = ''.join(str(e) for e in new_list)
clean_data = clean_data.split(".")
# clean_data=word_tokenize(clean_data)
print(clean_data)

nlp = spacy.load("en_core_web_sm")

tuples_list = []
for sentence in clean_data:
    val = nlp(sentence)
    tuples = textacy.extract.subject_verb_object_triples(val)
    if tuples:
        tuples_to_list = list(tuples)
        tuples_list.append(tuples_to_list)
        # print(tuples_list)
# print(tuples_list)

# Removing empty tuples in the list
final = []


def Remove(tuples):
    final = [t for t in tuples if t]
    return final


s_v_o = Remove(tuples_list)
# print(s_v_o)
s_v_o = ''.join(str(e) for e in s_v_o)
s_v_o = ''.join(str(e) for e in s_v_o)
print(s_v_o)