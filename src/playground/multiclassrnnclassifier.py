"""
Thanks to Susan Li for this step by step guide: https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../..')
from src.playground.feature_utils import get_emojis_from_text
sys.path.append('../../src/extern/indic_nlp_library/')
from src.extern.indic_nlp_library.indicnlp.normalize.indic_normalize import BaseNormalizer

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 50
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

def load_data(infile, mode, lb = None):
    df = pd.read_csv(infile, sep='\t')
    df.info()
    print(df.category.value_counts())
    df = df.reset_index(drop=True)
    
    tokenizer.fit_on_texts(df.text.values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df.text.values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)
    if lb is None:
        lb = LabelBinarizer()
        Y = lb.fit_transform(df.category.values.reshape(-1, 1))
    else:
        Y = lb.transform(df.category.values.reshape(-1, 1))
    print('Shape of label tensor:', Y.shape)
    return (X, Y, lb)

train_file = '../../resources/data/tamil_train.tsv'
X_train, Y_train, lb = load_data(train_file, 'train')
test_file = '../../resources/data/tamil_dev.tsv'
X_test, Y_test, lb = load_data(test_file, 'test', lb)
""" labelmap = {}
for category, y in zip(df.category.values, Y):
    # hack to reverse engineer the mapping
    labelmap[np.where(y==1)[0][0]] = category
    print(category, y)
    if len(labelmap.keys()) == Y.shape[1]:
        break
print(labelmap) """

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

""" import sys
sys.exit() """

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.7))
model.add(LSTM(150, dropout=0.5, recurrent_dropout=0.4))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

epochs = 8
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

new_review = ['Thalaiva superstar Rajinikanth number one mass Hero']
seq = tokenizer.texts_to_sequences(new_review)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
#labelid = 
print(pred, lb.inverse_transform(pred))