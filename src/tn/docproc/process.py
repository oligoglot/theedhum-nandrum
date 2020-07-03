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
from src.tn.docproc.emojitagger import EmojiTagger
from src.tn.docproc.languagetagger import LanguageTagger
from src.tn.docproc.spellchecktagger import SpellCheckTagger

if __name__ == "__main__":
    text = "woooood issssss your oyster.... 🥰 ###!!! ప్రపంచం అంతా వెతికిన ధనుష్ 🤩 లాంటి మరో నటుడు దొరకడు, 🤩 சுயமாக சிந்திக்க தெரிஞ்சவன் தான் சூப்பர் ஹீரோ 🥰 ಬಠಪಢಝ ಜಂಅಂಇ ಋಋ ಡಘಫಫಝ ಡಝಫಷ"
    text = "woooood issssss your oyester       .... 🥰 ###!!! சுயமாக சிந்திக்க 🤩 beer"
    doc = Document(text)
    pipeline = Pipeline()
    
    pipeline.addStep(Whitespace())
    pipeline.addStep(Lowercase())
    pipeline.addStep(Regexes())
    pipeline.addStep(EmojiTagger())
    pipeline.addStep(LanguageTagger())
    pipeline.addStep(SpellCheckTagger())
    pipeline.process(doc)