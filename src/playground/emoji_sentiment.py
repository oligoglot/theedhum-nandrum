import linecache
import sys
import emoji
import re
import csv
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../..') 

from  src.tn.lib.sentimoji import get_emoji_sentiment_rank

def extract_emojis(s):
    return [c for c in s if c in emoji.UNICODE_EMOJI]

matchedFn = "../../resources/data/matched_emojis.txt"
unmatchedFn = "../../resources/data/unmatched_emojis.txt"
matched = []
unmatched = []
# Get the list of unmatched emojis and put it in a file so that we can process it.
fileName = "../../resources/data/tamil_train.tsv"

with open (fileName, "r", encoding="UTF-8") as mainFile, open(matchedFn, "w", encoding="UTF-8") as matchedFile, open(unmatchedFn, "w", encoding="UTF-8") as unmatchedFile:
    readTsv = csv.reader(mainFile, delimiter="\t")
    for row in readTsv:
        txt, emotion = row
        emojiji = extract_emojis(txt)
        for em in emojiji:
            sentiment = get_emoji_sentiment_rank(em)
            if sentiment == False:
                if em not in unmatched:
                    unmatched.append(em)
            else:
                if em not in matched:
                    matched.append(em)

    matchedFile.writelines("\n".join(matched))
    unmatchedFile.writelines("\n".join(unmatched))
sys.exit()
'''
lineNum = 626
line = linecache.getline(fileName, lineNum)
text = line.split("\t")[0].strip()

emojiji = extract_emojis(text)
for em in emojiji:
    sentiment = get_emoji_sentiment_rank(em)
    print (sentiment)
'''