'''
@author mojosaurus
This step does a bunch of things as part of document cleanup. More details in comments below
'''
import sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../../..') 
from src.tn.docproc.pipeline import Step
from src.tn.document.document import Document
import re

class Regexes(Step):
    def __init__(self, document=Document()):
        self.document = document
        print ("Inside object of type : {}".format(self.__class__.__name__))
    
    # Replaces everything to lowercase, if it's latin alphabet.
    def execute(self):
        '''
        Note: Could have used a single regex like [r"(\.|\?|\!)+", "\\1"], but for some reason, the middle replacement is ignored
        '''
        regexes = [
            [r"(\.)+", "\\1"], # 1. Multiple fullstops with one fullstop
            [r"(\?)+", "\\1"], # 2. Multiple exclamation marks with one exclamation mark
            [r"(\!)+", "\\1"], # 3. Multiple question marks with one question mark
            [r"(#)+", "\\1"],  # 4. Multiple hash with one question mark
            [r"(\w)\1{2,}", "\\1\\1"],  # 5. Replaces multiple(2+) occurance of the same letter with 2 occurances.
        ]
        print ("Before processing : {} :  {}".format(self.__class__.__name__, self.document))
        for reg in regexes:
            self.document.set("text", re.sub(reg[0], reg[1], self.document.get("text")))
        print ("After processing : {} :  {}".format(self.__class__.__name__, self.document))