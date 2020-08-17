"""
Borrows from https://scikit-learn.org/0.18/auto_examples/hetero_feature_union.html
=============================================
Feature Union with Heterogeneous Data Sources
=============================================

Datasets can often contain components of that require different feature
extraction and processing pipelines.  This scenario might occur when:

1. Your dataset consists of heterogeneous data types (e.g. raster images and
   text captions)
2. Your dataset is stored in a Pandas DataFrame and different columns
   require different processing pipelines.

This example demonstrates how to use
:class:`sklearn.feature_extraction.FeatureUnion` on a dataset containing
different types of features.  We use the 20-newsgroups dataset and compute
standard bag-of-words features for the subject line and body in separate
pipelines as well as ad hoc features on the body. We combine them (with
weights) using a FeatureUnion and finally train a classifier on the combined
set of features.

The choice of features is not particularly helpful, but serves to illustrate
the technique.
"""

# Author: Matt Terry <matt.terry@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

import sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../..')
from src.playground.feature_utils import load_docs, get_emojis_from_text

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, reviews):
        return [{'length': len(text),
                 'num_sentences': text.count('.')}
                for text in reviews]


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract review text, emojis and emoji sentiment.

    Takes a sequence of strings and produces a dict of values.  Keys are
    `review`, `emojis`, and `emoji-sentiment`.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, reviews):
        features = np.recarray(shape=(len(reviews),),
                               dtype=[('review', object), ('emojis', object), ('emoji_sentiment', object)])
        for i, review in enumerate(reviews):       
            features['review'][i] = review

            emojis, sentiment = get_emojis_from_text(review)
            features['emojis'][i] = ' '.join(emojis)
            features['emoji_sentiment'][i] = sentiment

        return features


pipeline = Pipeline([
    # Extract the review text & emojis
    ('reviewfeatures', FeatureExtractor()),

    # Use FeatureUnion to combine the features from emojis and text
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's emojis
            # ('emojis', Pipeline([
            #     ('selector', ItemSelector(key='emojis')),
            #     ('vect', HashingVectorizer()),
            # ])),

            # Pipeline for standard bag-of-words model for review
            ('emojis', Pipeline([
                ('selector', ItemSelector(key='emojis')),
                ('tfidf', TfidfVectorizer(token_pattern=r'[^\s]+', stop_words=None, max_df=0.5, min_df=1)),
            ])),

            # Pipeline for pulling features from the post's emoji sentiment
            ('emoji_sentiment', Pipeline([
                ('selector', ItemSelector(key='emoji_sentiment')),
                ('vect', HashingVectorizer()),
            ])),

            # Pipeline for standard bag-of-words model for review
            ('review_bow', Pipeline([
                ('selector', ItemSelector(key='review')),
                ('tfidf', TfidfVectorizer( input='content', stop_words=None, sublinear_tf=True, max_df=0.5, min_df=1)),
                ('best', TruncatedSVD(n_components=50)),
            ])),

            # Pipeline for pulling ad hoc features from review text
            ('review_stats', Pipeline([
                ('selector', ItemSelector(key='review')),
                ('stats', TextStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

             # Pipeline for standard bag-of-words model for review
            ('review_ngram', Pipeline([
                ('selector', ItemSelector(key='review')),
                ('tfidf', CountVectorizer(ngram_range=(1, 3))),
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={ 
            'emoji_sentiment': 0.6,
            'emojis': 0.3,
            'review_bow': 1.0,
            'review_ngram': 0.5
        },
    )),

    # Use a SVC/SGD classifier on the combined features
    #('svc', SVC(kernel='linear')),
    ('sgd', SGDClassifier(loss="log", penalty="elasticnet",max_iter=70, random_state=0)),
])

# limit the list of categories to make running this example faster.
data_train = load_docs("../../resources/data/tamil_train.tsv")
data_test = load_docs("../../resources/data/tamil_dev.tsv")
print(data_train['data'][5], data_train['target_names'][4])
print('data loaded')
# order of labels in `target_names` can be different from `categories`
target_names = data_train['target_names']

pipeline.fit(data_train['data'], data_train['target_names'])
y = pipeline.predict(data_test['data'])
idx = 0
for v in data_test['data']:
  if (y[idx] == data_test['target_names'][idx]):
    print("Right : {} -> Prediction : {} -> Original : {}".format(v, y[idx], data_test['target_names'][idx]))
  else:
    print("Wrong : {} -> Prediction : {} -> Original : {}".format(v, y[idx], data_test['target_names'][idx]))
  idx += 1

print(classification_report(y, data_test['target_names']))
