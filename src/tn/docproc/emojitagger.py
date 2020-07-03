'''
@author mojosaurus
Tags emojis
'''
import sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../../..') 
from src.tn.document.document import Document
from src.tn.docproc.pipeline import Tagger
from src.tn.document.emojihelper import Emoji, EmojiHelper

class EmojiTagger(Tagger):
    def __init__(self, document=Document()):
        self.document = document
        self.helper = EmojiHelper()
        print ("Inside object of type : {}".format(self.__class__.__name__))
    
    # Replaces everything to lowercase, if it's latin alphabet.
    def execute(self):
        print ("Before processing : {} :  {}".format(self.__class__.__name__, self.document))

        tagged = self.helper.extractEmojiTags(self.document)
        self.document.set("tagged", tagged)
        #self.document.set("text", self.document.get("text").lower())
        print ("After processing : {} :  {}".format(self.__class__.__name__, self.document))