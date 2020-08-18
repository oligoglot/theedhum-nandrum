"""
Borrows from https://scikit-learn.org/0.18/auto_examples/hetero_feature_union.html
=============================================
Feature Union with Heterogeneous Data Sources
=============================================
"""

# Author: Matt Terry <matt.terry@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

import sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../..')
from src.playground.feature_utils import load_docs, get_emojis_from_text
sys.path.append('../../src/extern/indic_nlp_library/')
from src.extern.indic_nlp_library.indicnlp.normalize.indic_normalize import BaseNormalizer

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
    def __init__(self, lang = 'ta'):
        self.lang = lang
        self.normalizer = BaseNormalizer(lang)
        super().__init__()

    def fit(self, x, y=None):
        return self

    def transform(self, reviews):
        features = np.recarray(shape=(len(reviews),),
                               dtype=[('review', object), ('emojis', object), ('emoji_sentiment', object)])
        for i, review in enumerate(reviews):       
            features['review'][i] = self.normalizer.normalize(text = review)

            emojis, sentiment = get_emojis_from_text(review)
            features['emojis'][i] = ' '.join(emojis)
            features['emoji_sentiment'][i] = sentiment

        return features

def fit_predict_measure(mode, train_file, test_file, lang = 'ta'):
    print(train_file, test_file)
    data_train = load_docs(train_file, mode='train')
    data_test = load_docs(test_file, mode=mode)
    print('data loaded')
    target_names = data_train['target_names']

    pipeline = get_pipeline(lang)
    pipeline.fit(data_train['data'], data_train['target_names'])
    """ params = pipeline.get_params(deep=True)
    print(params['rsrch__estimator__alpha'], params['rsrch__estimator__penalty']) """
    y = pipeline.predict(data_test['data'])
    print(len(y))
    assert(len(data_test['data'])==len(y))
    if mode == 'test':
        idx = 0
        for v in data_test['data']:
            if (y[idx] == data_test['target_names'][idx]):
                print("Right : {} -> Prediction : {} -> Original : {}".format(v, y[idx], data_test['target_names'][idx]))
            else:
                print("Wrong : {} -> Prediction : {} -> Original : {}".format(v, y[idx], data_test['target_names'][idx]))
            idx += 1

        print(classification_report(y, data_test['target_names']))
    if mode == 'predict':
        with open(f'theedhumnandrum_{lang}.tsv', 'w') as outf:
            for idx, review, label in zip(data_test['ids'], data_test['data'], y):
                print(idx)
                outf.write('\t'.join((idx, review, label)) + '\n')
        print(f'predict data written to theedhumnandrum_{lang}.tsv')

def get_pipeline(lang = 'ta'):
    clf = SGDClassifier()

    """ distributions = dict(
        penalty=['l1', 'l2', 'elasticnet'],
        alpha=uniform(loc=1e-6, scale=1e-4)
    ) """

    pipeline = Pipeline([
        # Extract the review text & emojis
        ('reviewfeatures', FeatureExtractor(lang)),

        # Use FeatureUnion to combine the features from emojis and text
        ('union', FeatureUnion(
            transformer_list=[

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

        # Use an SVC/SGD classifier on the combined features
        #('svc', SVC(kernel='linear')),
        ('sgd', SGDClassifier(loss="log", penalty="elasticnet", max_iter=500, random_state=0)),
        # ('rsrch', RandomizedSearchCV(estimator=clf, param_distributions=distributions, cv=5, n_iter=5)),
    ])
    return pipeline

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 5:
        print('Your command should be:')
        print('python hetero_feature_union.py <mode> <language code> <training file path> <test file path>')
        print('mode:predict/test, language: ta/ml')
        sys.exit()
    
    mode, lang, train_file, test_file = args[1:5]
    fit_predict_measure(mode, train_file, test_file, lang = lang)