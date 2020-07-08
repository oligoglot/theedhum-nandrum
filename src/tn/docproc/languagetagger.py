'''
@author mojosaurus
Tags languages
'''

import os,sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append(os.path.join(os.path.dirname(__file__),'../../..'))
from src.tn.document.document import Document
from src.tn.docproc.pipeline import Tagger
from src.tn.document.languagehelper import LanguageHelper
import cld2


class LanguageTagger(Tagger):
    def __init__(self, document=Document()):
        self.document = document
        self.helper = LanguageHelper()
        print ("Inside object of type : {}".format(self.__class__.__name__))
    
    # Uses CLD2 library to identify and tags part of a multi-script sentence.
    # TODO: Replace CLD2 with CLD3.
    def execute(self):
        print ("Before processing : {} :  {}".format(self.__class__.__name__, self.document))
        tagged = self.helper.extractLanguageTags(self.document)
        self.document.set("tagged", tagged)
        print ("After processing : {} :  {}".format(self.__class__.__name__, self.document))