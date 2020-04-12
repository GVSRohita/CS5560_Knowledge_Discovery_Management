import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

def extract_features(word_list):
    return dict([(word, True) for word in word_list])

if __name__=='__main__':
   # Load positive and negative reviews
   positive_fileids = movie_reviews.fileids('pos')
   negative_fileids = movie_reviews.fileids('neg')
   features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                         'Positive') for f in positive_fileids]
   features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                         'Negative') for f in negative_fileids]

   # Split the data into train and test (80/20)
   threshold_factor = 0.8
   # threshold_factor = 0.7
   # threshold_factor = 0.5
   threshold_positive = int(threshold_factor * len(features_positive))
   threshold_negative = int(threshold_factor * len(features_negative))

   features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
   features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
   print("\nNumber of training datapoints:", len(features_train))
   print("Number of test datapoints:", len(features_test))

   # Train a Naive Bayes classifier
   classifier = NaiveBayesClassifier.train(features_train)
   print("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))

   print("\nTop 10 most informative words:")
   for item in classifier.most_informative_features()[:10]:
        print(item[0])

    # Sample input reviews
   input_reviews = [
       "It is an amazing movie",
       "This is a dull movie. I would never recommend it to anyone.",
       "The cinematography is pretty great in this movie",
       "The direction was terrible and the story was all over the place"
   ]
   # input_reviews = [
   #     "This was easy to put together. It is sturdy ad a perfect fit for my daughter's room. We have one drawer that sticks a little COMMA  but it works.",
   #     "I loved the look of this dresser in the store and decided to take the plunge. It was a major project to assemble (6-ish hours for one relatively handy person without power tools) COMMA  but the finished product looks great and stores a ton of clothes! The directions could definitely be a little clearer on assembling the middle divider pieces COMMA  which looks wrong even when done correctly and the dimples in the wood for orientation look like holes in the instructions. I couldn't get two of the four screws that connect the front face to the dresser top to go in (screws too short or holes not quite aligned) COMMA  but thankfully there were many other points of attachment and it's not at all obvious that they're missing. And mine came with metal (not plastic) cam locks COMMA  which is a good thing. Great buy!",
   #     "We were very disappointed to realize that the hemnes set in white is made of mostly particle board. We were under the impression that the all the hemnes line was made of solid wood. After further investigation it seems as though all the dressers are made of wood except the white ones. Not sure why this is and is very misleading",
   #     "I not only purchased the dresser but I bought the matching chest I'd drawers. The pieces took a while to put together but they are worth the time. Great product."
   # ]
   print("\nPredictions:")
   for review in input_reviews:
       print("\nReview:", review)
       probdist = classifier.prob_classify(extract_features(review.split()))
       pred_sentiment = probdist.max()
       print("Predicted sentiment:", pred_sentiment)
       print("Probability:", round(probdist.prob(pred_sentiment), 2))



