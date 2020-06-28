# Load and prepare the dataset
import nltk
from nltk.corpus import movie_reviews
import random
import sys
import re

# documents = [(list(movie_reviews.words(fileid)), category)
#               for category in movie_reviews.categories()
#               for fileid in movie_reviews.fileids(category)]
documents = []
f = sys.argv[1]
with open(f) as inf:
    for line in inf:
        (review, cat) = re.split('\t', line.strip())
        words = review.split()
        document = (list(words), cat)
        documents.append(document)
random.shuffle(documents)

# Define the feature extractor

imdb_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
imdb_words = list(imdb_words)[:2000]
training_words = nltk.FreqDist(w.lower() for d in documents for w in d[0])
training_words = list(training_words)[:2000]
all_words = imdb_words + training_words
word_features = all_words

def document_features(document):
    document_words = set(document)
    # TODO: use bigrams in both training and testing
    #document_bigrams = set(list(nltk.bigrams(document)))
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

test_size = int(len(documents)/20.0)
# Train Naive Bayes classifier
featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[test_size:], featuresets[:test_size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test the classifier
print(nltk.classify.accuracy(classifier, test_set))
