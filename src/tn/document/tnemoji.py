'''
@author mojosaurus
Theedhum Nandrum version of emoji class
'''
import emoji
from  src.tn.document.collectiontuple import CollectionTuple

# Emoji is a special case of CollectionTuple
class Emoji(CollectionTuple):
    def __init__(self, lang="emoji", text=""):
        # jiji is a random name to capture our version of emoji
        self.jiji = {
            "lang"      : lang,
            "text"      : text,
            "unicodeName"       : 0,
            "unicodeBlock"      : 0
        }

# This is the helper class for emojis in the context of theedhum nandrum.
# All emoji related functions go here.
class EmojiHelper:
    def __init__(self):
        pass

    # Given a document, this function extracts all the emojis  
    def extractEmojis(self, document):
        # Iterate over each of the tagged portions of the document to identify the emojis
        absolutePos = 0
        taggedIndex = 0
        
        for tagged in document.get("tagged"):
            currCollection = []
            collection = []
            relativePos = 0
            for c in tagged:
                # Iterate over each character to find if this charcter is an emoji
                if c in emoji.UNICODE_EMOJI:
                    print ("{} is an emoji".format(c))
                    # Now that we have found an emoji, close currCollection
                    text="".join(currCollection)
                    if text != "":
                        jiji = CollectionTuple(text=text)
                        jiji.set("relativePos", relativePos)
                        jiji.set("absolutePos", absolutePos)
                        collection.append(jiji.getJiji())
                        currCollection = []
                        relativePos += len(text)
                        absolutePos += len(text)

                    # Now add the emoji to collection
                    jiji = Emoji(lang="emoji", text=c)
                    jiji.set("relativePos", relativePos)
                    jiji.set("absolutePos", absolutePos)
                    collection.append(jiji.getJiji())
                    relativePos += 1
                    absolutePos += 1
                    # TODO: Add the rest of the attributes

                else:
                    # Keep adding to the current row in collection
                    currCollection.append(c)
                #relativePos += 1
                #absolutePos += 1
            text="".join(currCollection)
            if text != "":
                jiji = CollectionTuple(text=text)
                jiji.set("relativePos", relativePos)
                jiji.set("absolutePos", absolutePos)
                collection.append(jiji.getJiji())
                
            # TODO: Replace the current tagged position with jijiCollection and increment the taggedIndex.
            print ("Collection is {}".format(collection))
            # TODO: Replace the current tagged Index with this collection
            taggedIndex += 1

        #return [c for c in text if c in emoji.UNICODE_EMOJI]