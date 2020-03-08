# Text processing
# Import library
import re
import string
import nltk
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np

# Read input of headlines of ABC News of text format from a csv file
reviews_df=pd.read_csv('Womens Clothing E-Commerce Reviews.csv',error_bad_lines=False)
reviews_df['Reviews'] = reviews_df['Review Text'].astype(str)
print(reviews_df.head(6))

# Cleaning data as pre-processing step along with converting the entire data into lowercase
def initial_clean(text):
    """
    Function to clean text-remove punctuations, lowercase text etc.
    """
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower()  # lower case text
    text = nltk.word_tokenize(text)
    return (text)

# Processing and removing stop words holding no significance towards Topic identification
# Identifying and importing stop words from NLTK library
stop_words = stopwords.words('english')
stop_words.extend(['news', 'say','use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do','took','time','year',
'done', 'try', 'many', 'some','nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line','even', 'also', 'may', 'take', 'come', 'new','said', 'like','people'])
# Removing stop words identified from the library and the list provided by the user
def remove_stop_words(text):
     return [word for word in text if word not in stop_words]

# Stemming of words - removing plurals and adjectives to derive base word
stemmer = PorterStemmer()
def stem_words(text):
    """
    Function to stem words
    """
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] # no single letter words
    except IndexError:
        pass

    return text

# Apply all functions defined above on the text when executed
def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return stem_words(remove_stop_words(initial_clean(text)))


# Clean data and create new column "tokenized"
# Identify time taken to clean and tokenize data
import time
t1 = time.time()
reviews_df['tokenized_reviews'] = reviews_df['Reviews'].apply(apply_all)
t2 = time.time()
print("Time to clean and tokenize", len(reviews_df), "reviews:", (t2-t1)/60, "min") #Time to clean and tokenize 3209 reviews: 0.21254388093948365 min

print('\n')
print("reviews with their respective tokenize version:" )
print(reviews_df.head(5))

# Latent Dirichlet Aloocation (LDA) Analysis
# Import library
import gensim
import pyLDAvis.gensim
# Create a gensim dictionary from the tokenized data
tokenized = reviews_df['tokenized_reviews']
# Create term dictionary of corpus, where each unique term is assigned an index.
dictionary = corpora.Dictionary(tokenized)
# Filter terms which occurs in less than 1 review and more than 80% of the reviews.
dictionary.filter_extremes(no_below=1, no_above=0.8)
# Convert dictionary into a bag-of-words corpus
corpus = [dictionary.doc2bow(tokens) for tokens in tokenized]
# Display corpus of words
print(corpus[:1])
# Display dictionary identified and derived
print([[(dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]])

# Latent Diichlet Allocation (LDA) Model
# Generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5, id2word=dictionary, passes=15)
# Save the model generated for future use
ldamodel.save('model_combined.gensim')

# Identifying topics from the corpus of data
topics = ldamodel.print_topics(num_words=5)
print('\n')
# Displaying topics identified along with their composition elements
print("Now printing the topics and their composition")
print("This output shows the Topic-Words matrix for the 7 topics created and the 4 words within each topic")
for topic in topics:
   print(topic)

# Analysing & determining similarity score of the first review associated with topics identified
print('\n')
print("first review is:")
print(reviews_df.Reviews[0])
get_document_topics = ldamodel.get_document_topics(corpus[0])
print('\n')
print("The similarity of this review with the topics and respective similarity score are ")
print(get_document_topics)

# Visualization of topics identified using LDA
lda_viz = gensim.models.ldamodel.LdaModel.load('model_combined.gensim')
lda_display = pyLDAvis.gensim.prepare(lda_viz, corpus, dictionary, sort_topics=True)
pyLDAvis.show(lda_display)
