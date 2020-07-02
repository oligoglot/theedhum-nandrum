'''
@author mojosaurus
Language helper class
'''

from  src.tn.document.collectiontuple import CollectionTuple
import cld2

# This is the helper class for language classification in the context of theedhum nandrum.
# All languahe related functions go here.
class LanguageHelper:
    def __init__(self):
        pass

    # Given a document, this function extracts all the emojis  
    def extractLanguageTags(self, document):
        # Iterate over each of the tagged portions of the document to identify the languages
        absolutePos = 0
        taggedIndex = 0
        collection = []
        for tagged in document.get("tagged"):
            if tagged["lang"] == "emoji":
                collection.append(tagged)
                continue
            
            text = tagged["text"]
            isReliable, textBytesFound, details, vectors = cld2.detect(text, returnVectors=True)

            #print('  reliable: %s' % (isReliable != 0))
            #print('  textBytes: %s' % textBytesFound)
            #print('  details: %s' % str(details))
            i=0
            for vector in vectors:
                print ("*************")
                #print (vector)
                print (details[i])
                start = vector[0]
                end = vector[1]
                print ("Start : {}, end : {}".format(start, start+end))
                print (vector)
                print (text[start:start+end])

                jiji = CollectionTuple(text=text[start:start+end])
                jiji.set("relativePos", start)
                jiji.set("absolutePos", start)
                jiji.set("lang", vector[3])
                collection.append(jiji.getJiji())

                i += 1
                print ("*************")
        return collection