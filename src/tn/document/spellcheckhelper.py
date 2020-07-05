'''
@author mojosaurus
Spellcheck helper class
'''

from  src.tn.document.collectiontuple import CollectionTuple
import src.tn.lib.spell as spell
from src.tn.document.document import Document


# This is the helper class for spell check and correction in the context of theedhum nandrum.
# Currently it takes handles only english spell check and correction.
# All spell-check and spell correct related functions go here.
class SpellCheckHelper:
    def __init__(self):
        pass

    # Given a document, this function tries to correct incorrect spellings  
    def correct(self, document:Document):
        # Iterate over each of the tagged portions of the document
        # Note that language tagging should have been done before this step.
        collection = []
        for tagged in document.get("tagged"):
            if tagged["lang"] == "emoji":
                collection.append(tagged)
                continue
            if tagged["lang"] == "un":
                text = tagged["text"]
                corrected = []
                for word in text.split(" "):
                    # TODO: More thoughts need to get into this.
                    cword = spell.correction(word)
                    print ("Original : {}, corrected : {}".format(word, cword))
                    corrected.append(cword)
                
                jiji = CollectionTuple(text=" ".join(corrected))
                collection.append(jiji.getJiji())
            else:
                # Other languages go here. It's empty for now.
                collection.append(tagged)
                continue
        return collection