# -*- coding: utf-8 -*-
import snowballstemmer
from stemmer import Stemmer

class TamilStemmer(Stemmer):
  def __init__(self):
    self.stemmer = snowballstemmer.stemmer("tamil")
    self.text = "hello"

  def stemword(self, input):
    return self.stemmer.stemWord(input)

s = TamilStemmer()
resp = s.stemword("பார்வைகள்")
print(resp)