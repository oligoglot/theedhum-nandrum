# Load and prepare the dataset
import nltk
from nltk.corpus import movie_reviews
import random
import sys
import re
from emoji import UNICODE_EMOJI
from bisect import bisect_left
import math

nltk.download('movie_reviews')
# documents = [(list(movie_reviews.words(fileid)), category)
#               for category in movie_reviews.categories()
#               for fileid in movie_reviews.fileids(category)]
documents = []
# f = sys.argv[1]
f = "../../resources/data/tamil_train.tsv"
with open(f, 'r', encoding='utf-8') as inf:
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
all_word_features_base = {}
for word in word_features:
  all_word_features_base['contains({})'.format(word)] = (False)
all_emoji_features_base = {}
for c in UNICODE_EMOJI:
  all_emoji_features_base['has-emoji({})'.format(c)] = (False)

def document_features(document, feature_sets):
    document_words = set(document)
    # TODO: use bigrams in both training and testing
    # document_bigrams = set(list(nltk.bigrams(document)))
    features = {}
    if ('occurance' in feature_sets):
      features.update(all_word_features_base)
      for word in document_words:
        features =  all_word_features_base.copy()
        features['contains({})'.format(word)] = (True)

    if ('emojis' in feature_sets):
        document_emoji_feature(document_words, features)

    if ('length' in feature_sets):
        document_length_feature(document_words, features)

    return features


def document_emoji_feature(document_words, features):
    features.update(all_emoji_features_base)
    allchars = set(''.join(document_words))
    for c in allchars:
        features['has-emoji({})'.format(c)] = (True)


def document_length_feature(document_words, features):
    #  features['word-count'] = len(document_words)
    doclen = sum(len(word) for word in document_words)
    features['doc-length'] = get_range(doclen)
    # features['avg-word-length'] = int(round(features['doc-length']/len(document_words)))


def get_range(doclen):
    ranges = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100", "101-110", "111-120", "121-130", "131-140",
              "141-150", "151-160", "161-170", "171-180", "181-190", "191-200", ">200"]
    breakpoints = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                   110, 120, 130, 140, 150, 160, 170, 180, 190, math.inf]
    index = bisect_left(breakpoints, doclen)
    return ranges[index]


test_size = int(len(documents)/20.0)
# Train Naive Bayes classifier
# featuresets = [(document_features(d, {'occurance'}), c)
#                for (d, c) in documents]
# train_set, test_set = featuresets[test_size:], featuresets[:test_size]
# classifier = nltk.NaiveBayesClassifier.train(train_set)

# # Test the classifier
# print(nltk.classify.accuracy(classifier, test_set))
# # classifier.show_most_informative_features(25)

# featuresets = [(document_features(d, {'emojis'}), c) for (d, c) in documents]
# train_set, test_set = featuresets[test_size:], featuresets[:test_size]
# classifier = nltk.NaiveBayesClassifier.train(train_set)

# # Test the classifier
# print(nltk.classify.accuracy(classifier, test_set))
# # classifier.show_most_informative_features(25)


# featuresets = [(document_features(d, {'length'}), c) for (d, c) in documents]
# train_set, test_set = featuresets[test_size:], featuresets[:test_size]
# classifier = nltk.NaiveBayesClassifier.train(train_set)

# # Test the classifier
# print(nltk.classify.accuracy(classifier, test_set))
# # classifier.show_most_informative_features(25)

featuresets = [
    (document_features(d, {'emojis'}), c) for (d, c) in documents]
train_set, test_set = featuresets[test_size:], featuresets[:test_size]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test the classifier
print(nltk.classify.accuracy(classifier, test_set))

#Classify a few docs and check
for(d, c) in documents[:100]:
    guess = classifier.classify(document_features(
        d, {'emojis'}))
    if(guess != c):
        print('Got It Wrong correct={} guess={} comment={}'.format(c, guess, ' '.join(d)))
    else:
        print('Got It Right guess={} comment={}'.format(guess, ' '.join(d).strip()))
