import linecache
import sys
import emoji
import re
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../..') 

from  src.tn.lib.sentimoji import get_emoji_sentiment_rank


def extract_emojis(s):
    return [c for c in s if c in emoji.UNICODE_EMOJI]

fileName = "resources/data/tamil_train.tsv"
lineNum = 614
line = linecache.getline(fileName, lineNum)
text = line.split("\t")[0].strip()

emojiji = extract_emojis(text)
for em in emojiji:
    sentiment = get_emoji_sentiment_rank(em)
    print (sentiment)
