'''
@author mojosaurus
This is a logical single block of the document 
'''

class CollectionTuple:
    def __init__(self, lang="", text=''):
        # jiji is a random name to capture our version of emoji
        self.jiji = {
            "lang"      : lang,
            "text"      : text,
            "negative"  : 0,
            "neutral"   : 0,
            "positive"  : 0,
            "sentimentScore"    : 0,  
        }

    def getJiji(self):
        return [self.jiji]

    def set(self, key, value):
        self.jiji[key] = value