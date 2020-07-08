'''
@author mojosaurus
Replaces multiple whitespaces with one whitespace
'''
import os,sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append(os.path.join(os.path.dirname(__file__),'../../..'))
from src.tn.docproc.pipeline import Step
from src.tn.document.document import Document

class Whitespace(Step):
    def __init__(self, document=Document()):
        self.document = document
        print ("Inside object of type : {}".format(self.__class__.__name__))
    
    # Replaces multiple whitespaces with one whitespace.
    def execute(self):
        print ("Before processing : {} :  {}".format(self.__class__.__name__, self.document))
        self.document.set("text", ' '.join(self.document.get("text").split()))
        print ("After processing : {} :  {}".format(self.__class__.__name__, self.document))