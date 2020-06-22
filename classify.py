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
        print(line)
        (review, cat) = re.split('\t', line.strip())
        words = review.split()
        document = (list(words), cat)
        documents.append(document)
random.shuffle(documents)

# Define the feature extractor

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
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
