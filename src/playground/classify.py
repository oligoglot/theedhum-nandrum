# Load and prepare the dataset
import nltk
from nltk.corpus import movie_reviews
from nltk.util import ngrams
import random
import sys
import re
from emoji import UNICODE_EMOJI
from bisect import bisect_left
import math
from sklearn.metrics import classification_report
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../..')
from  src.tn.lib.sentimoji import get_emoji_sentiment_rank

nltk.download('movie_reviews')
#nltk_documents = [(list(movie_reviews.words(fileid)), category)
#               for category in movie_reviews.categories()
#               for fileid in movie_reviews.fileids(category)]
def load_docs(source):
    documents = []
    with open(source, 'r', encoding='utf-8') as inf:
        for line in inf:
            (review, cat) = re.split('\t', line.strip())
            words = review.split()
            document = (list(words), cat)
            documents.append(document)
    return documents


# Define the feature extractor

def document_features(document, feature_sets):
    document_words = set(document)
    # TODO: use bigrams in both training and testing
    # document_bigrams = set(list(nltk.bigrams(document)))
    features = {}
    if ('bag_of_words' in feature_sets):
        document_bag_of_words_feature(document_words, features)

    if ('emojis' in feature_sets):
        document_emoji_feature(document_words, features)

    if ('length' in feature_sets):
        document_length_feature(document_words, features)

    if ('ngram' in feature_sets):
        for size in feature_sets['ngram']:
            document_ngram_feature(document, features, size)
    return(features)


def get_bag_of_all_words():
    if not hasattr(get_bag_of_all_words, "bag_of_words"):
        get_bag_of_all_words.bag_of_words = {}
        imdb_words = list(nltk.FreqDist(w.lower()
                                        for w in movie_reviews.words()))[:2000]
        training_words = nltk.FreqDist(w.lower()
                                       for d in training_documents for w in d[0])
        training_words = list(training_words)[:2000]
        all_words = imdb_words + training_words
        word_features = all_words
        for word in word_features:
            get_bag_of_all_words.bag_of_words['contains({})'.format(
                word)] = (False)
    return get_bag_of_all_words.bag_of_words

# The bag of Words Feature Classifier. Marks occurance of words from the universal
# dictonary


def document_bag_of_words_feature(document_words, features):
    bag_of_words = get_bag_of_all_words()
    features.update(bag_of_words)
    for word in document_words:
        features['contains({})'.format(word)] = (True)


def get_all_emojis():
    if not hasattr(get_all_emojis, "all_emojis"):
        get_all_emojis.all_emojis = {}
        for c in UNICODE_EMOJI:
            get_all_emojis.all_emojis['has-emoji({})'.format(c)] = (False)
    return get_all_emojis.all_emojis


# The emoji feature classifier
def document_emoji_feature(document_words, features):
    all_emojis = get_all_emojis()
    features.update(all_emojis)
    allchars = set(''.join(document_words))
    score = 0.0
    for c in allchars:
        features['has-emoji({})'.format(c)] = (True)
        sentiment = get_emoji_sentiment_rank(c)
        if sentiment is not False:
            score += sentiment['sentiment_score']
    features['emoji-positive'] = (False)
    features['emoji-negative'] = (False)
    features['emoji-neutral'] = (False)
    if score > 0.2:
        features['emoji-positive'] = (True)
    elif score < -0.2:
        features['emoji-negative'] = (True)
    else:
        features['emoji-neutral'] = (True)



def document_length_feature(document_words, features):
    features['word-count'] = len(document_words)
    # doclen = sum(len(word) for word in document_words)
    # features['doc-length'] = get_range(doclen)
    # features['avg-word-length'] = int(round(features['doc-length']/len(document_words)))


def get_range(doclen):
    ranges = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100", "101-110", "111-120", "121-130", "131-140",
              "141-150", "151-160", "161-170", "171-180", "181-190", "191-200", ">200"]
    breakpoints = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                   110, 120, 130, 140, 150, 160, 170, 180, 190, math.inf]
    index = bisect_left(breakpoints, doclen)
    return ranges[index]

# Similar to bag of words filter, but for N grams


def get_all_ngrams(n):
    if not hasattr(get_all_ngrams, "all_ngrams"):
        get_all_ngrams.all_ngrams = {}
        imdb_ngrams = list(ngrams(movie_reviews.words(), n))[:2000]
        training_ngrams = []
        for d in training_documents:
            training_ngrams.extend(ngrams(d[0], n))
        training_ngrams = training_ngrams[:2000]
        total_ngrams = imdb_ngrams + training_ngrams
        for ngram in total_ngrams:
            get_all_ngrams.all_ngrams['contains({})'.format(
                "-".join(ngram))] = (False)
    return get_all_ngrams.all_ngrams


def document_ngram_feature(doc, features, n):
    all_ngrams = get_all_ngrams(n)
    doc_ngrams = list(ngrams(doc, n))
    features.update(all_ngrams)
    for ngram in doc_ngrams:
        features['contains({})'.format("-".join(ngram))] = (True)

# Output classification in sklearn report format - 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
def get_classifier_metrics_report(classifier, inputset, features):
  refset, guesset= [], []
  for (d,c) in inputset:
    refset.append(c)
    guesset.append(classifier.classify(document_features(d, features)))
  return classification_report(refset, guesset)

training_documents = load_docs("../../resources/data/tamil_train.tsv")
testing_documents = load_docs("../../resources/data/tamil_dev.tsv")
# random.shuffle(documents)
# test_size = int(len(documents)/20.0)

feature_filters = [{'length': 1}, {'bag_of_words': 1}, {'ngram': [4]}, {'ngram': [5]}, {
    'length': 1, 'ngram': [5]}, {'length': 1, 'ngram': [4]}, {'emojis': 1}, {'emojis': 1, 'ngram': [2, 3, 4]},
    {'bag_of_words': 1, 'ngram': [2, 3, 4], 'length': 1, 'emojis': 1}]
for filter in feature_filters:
    # Train Naive Bayes classifier
    train_set = [
        (document_features(d, filter), c) for (d, c) in training_documents]
    test_set = train_set = [
        (document_features(d, filter), c) for (d, c) in training_documents]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    report = get_classifier_metrics_report(classifier, test_set, filter)
    print("Classification report for classifier %s\n"
      % (report))
    # Test the classifier
    # print("{} -> {}". format(str(filter),
    #                          nltk.classify.accuracy(classifier, test_set)))

# Classify a few docs and check
# for(d, c) in documents[:100]:
#     guess = classifier.classify(document_features(
#         d, {'length' : 1 ,'ngram': 4}))
#     if(guess != c):
#         print('Got It Wrong correct={} guess={} comment={}'.format(
#             c, guess, ' '.join(d)))
#     else:
#         print('Got It Right guess={} comment={}'.format(
#             guess, ' '.join(d).strip()))
