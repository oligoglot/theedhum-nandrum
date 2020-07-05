'''
@author mojosaurus
This is a logical single block of the document 
'''

class CollectionTuple:
    def __init__(self, lang : str = "", text : str =""):
        # jiji is a random name to capture the context of the tuple. Too sleepy to name it anything. 
        self.jiji = {
            "lang"      : lang,
            "text"      : text,
            "negative"  : 0,
            "neutral"   : 0,
            "positive"  : 0,
            "sentimentScore"    : 0,  
        }

    def getJiji(self):
        return self.jiji

    def set(self, key : str, value : str):
        self.jiji[key] = value