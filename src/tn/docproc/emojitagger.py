'''
@author mojosaurus
Replaces multiple whitespaces with one whitespace
'''
import sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../../..') 
from src.tn.document.document import Document
from src.tn.docproc.pipeline import Tagger
from src.tn.document.tnemoji import Emoji, EmojiHelper

class EmojiTagger(Tagger):
    def __init__(self, document=Document()):
        self.document = document
        self.helper = EmojiHelper()
        print ("Inside object of type : {}".format(self.__class__.__name__))
    
    # Replaces everything to lowercase, if it's latin alphabet.
    def execute(self):
        print ("Before processing : {} :  {}".format(self.__class__.__name__, self.document))
        # 1. Identify the list of emojis
        tagged = self.helper.extractEmojis(self.document)
        self.document.set("tagged", tagged)
        #self.document.set("text", self.document.get("text").lower())
        print ("After processing : {} :  {}".format(self.__class__.__name__, self.document))