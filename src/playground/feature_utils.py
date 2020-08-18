from emoji import UNICODE_EMOJI
import re
import sys
from googletrans import Translator
from langdetect import detect
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../..')
from  src.tn.lib.sentimoji import get_emoji_sentiment_rank

def load_docs(source):
    documents = {'data': [], 'target_names': []}
    with open(source, 'r', encoding='utf-8') as inf:
        # skipping header row
        next(inf)
        for line in inf:
            (review, cat) = re.split('\t', line.strip())
            documents['data'].append(review)
            documents['target_names'].append(cat)
    return documents
    
def get_all_emojis():
    if not hasattr(get_all_emojis, "all_emojis"):
        get_all_emojis.all_emojis = {}
        for c in UNICODE_EMOJI:
            get_all_emojis.all_emojis['has-emoji({})'.format(c)] = (False)
    return get_all_emojis.all_emojis


# The emoji feature classifier
def document_emoji_feature(document_words, features):
    all_emojis = get_all_emojis()
    features.update(all_emojis)
    allchars = set(''.join(document_words))
    score = 0.0
    emojis = []
    for c in allchars:
        if c in UNICODE_EMOJI:
            emojis.append(c)
            features['has-emoji({})'.format(c)] = (True)
            sentiment = get_emoji_sentiment_rank(c)
            if sentiment is not False:
                score += sentiment['sentiment_score']
    features['emoji-positive'] = (False)
    features['emoji-negative'] = (False)
    features['emoji-neutral'] = (False)
    if len(emojis) > 0:
        score /= len(emojis)
    if score > 0.2:
        features['emoji-positive'] = (True)
    elif score < -0.2:
        features['emoji-negative'] = (True)
    else:
        features['emoji-neutral'] = (True)

def get_emojis_from_text(text):
    score = 0.0
    # Putting in a random emoji to avoid empty data
    emojis = ["🦻"]
    for c in text:
        if c in UNICODE_EMOJI:
            emojis.append(c)
            sentiment = get_emoji_sentiment_rank(c)
            if sentiment is not False:
                score += sentiment['sentiment_score']
    if len(emojis) > 0:
        score /= len(emojis)
    if score > 0.2:
        label = 'Positive'
    elif score < -0.2:
        label = 'Negative'
    else:
        label = 'Neutral'
    return ((emojis, label))

def get_language(text):
      #translator = Translator()
      try:
        return(detect(text))
      except:
        return("unknown")
      # if (language.confidence > 0.7): return language.lang
      # return "unknown"

def detect_lang_and_store(input, outputfile):
  with open(outputfile, "w") as f:
    for text in input:
      # Intentional re-init of object - https://stackoverflow.com/questions/49497391/googletrans-api-error-expecting-value-line-1-column-1-char-0
      translator = Translator()
      try:
        language = translator.detect(text)
        f.write(text + "\t" + language.lang + "\t" + str(language.confidence) + "\n")
      except Exception as e:
        print(str(e))
        continue
  f.close()



if __name__ == "__main__":
    # features = {}
    # document_words = 'ugh 🤢'
    # document_emoji_feature(document_words, features)
    # print(features)
    # document_words = 'கலக்கல் 🤩'
    # document_emoji_feature(document_words, features)
    # print(features)
    detect_lang_and_store(["idhu enna maayam", "sundari kannaal oru sedhi", "malalayali aano", "கலக்கல்", "nandri hai"], "/tmp/languages_tmp.tsv")