'''
@author mojosaurus
Replaces multiple whitespaces with one whitespace
'''
import os,sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append(os.path.join(os.path.dirname(__file__),'../../..'))
from src.tn.docproc.pipeline import Step
from src.tn.document.document import Document

class Lowercase(Step):
    def __init__(self, document=Document()):
        self.document = document
        print ("Inside object of type : {}".format(self.__class__.__name__))
    
    # Replaces everything to lowercase, if it's latin alphabet.
    def execute(self):
        print ("Before processing : {} :  {}".format(self.__class__.__name__, self.document))
        self.document.set("text", self.document.get("text").lower())
        print ("After processing : {} :  {}".format(self.__class__.__name__, self.document))