'''
@author mojosaurus
This file defines docproc pipeline to clean up input data.
This might be broken down into multiple intermediary steps and the final output comes out at last.
'''

'''
This baseclass defines the list of functions that need to be implemented in each of the steps in the
pipeline.
Each step in the pipeline expects the input and output in a certain tuple format defined as JSON.
TODO: Add a reference to the tuple 
'''
# This is the base class that needs to be inherited for all Steps, except Taggers
class Step:
    def __init__(self, document):
        print ("In baseclass")
        self.document = document

    # Inherited classes need to implement this method
    def execute(self):
        pass

    # Do not implement this method in inherited class
    def getDocument(self):
        return self.document

    # Do not implement this method in inherited class
    def setDocument(self, document):
        self.document = document 

# This is the base class that needs to be inherited for all Taggers in docproc pipeline
class Tagger(Step):
    def __init__(self):
        "Inside baseclass Tagger"


class Pipeline:
    pipelineSteps = []
    document = {}

    def __init__(self):
        pass

    # Expects an object of type Step to be passed
    def addStep(self, step):
        self.pipelineSteps.append(step)

    # This function iterates over the steps and executes
    def process(self, document):
        for step in self.pipelineSteps:
            #print (step.getDocument())
            step.setDocument(document)
            step.execute()
            document = step.getDocument()