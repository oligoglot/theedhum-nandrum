'''
@author mojosaurus
This is the file that executes the pipeline
'''
import sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../../..') 

from src.tn.docproc.pipeline import Step
from src.tn.docproc.whitespace import Whitespace
from src.tn.docproc.lowercase import Lowercase
from src.tn.document.document import Document
from src.tn.docproc.regexes import Regexes
from src.tn.docproc.pipeline import Pipeline

if __name__ == "__main__":
    text = "Hellooooo ####  !!!!!????? Cruellllll   Worrrrrld  "
    doc = Document(text)
    pipeline = Pipeline()
    
    pipeline.addStep(Whitespace())
    pipeline.addStep(Lowercase())
    pipeline.addStep(Regexes())
    pipeline.process(doc)