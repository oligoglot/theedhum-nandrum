import linecache
import sys
import emoji
import re
import csv
from collections import Counter
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../..') 

from  src.tn.lib.sentimoji import get_emoji_sentiment_rank

def extract_emojis(s):
    return [c for c in s if c in emoji.UNICODE_EMOJI]

matchedFn = "../../resources/data/matched_emojis.txt"
unmatchedFn = "../../resources/data/unmatched_emojis.txt"
matched = Counter()
unmatched = Counter()
occurences = Counter()
# Get the list of unmatched emojis and put it in a file so that we can process it.
fileName = "../../resources/data/all_records.tsv"

with open (fileName, "r", encoding="UTF-8") as mainFile, open(matchedFn, "w", encoding="UTF-8") as matchedFile, open(unmatchedFn, "w", encoding="UTF-8") as unmatchedFile:
    readTsv = csv.reader(mainFile, delimiter="\t")
    for row in readTsv:
        if len(row) == 2:
            txt, emotion = row
            emojiji = extract_emojis(txt)
            for em in emojiji:
                sentiment = get_emoji_sentiment_rank(em)
                if sentiment == False:
                    unmatched[(em,emotion)] += 1
                    occurences[em] += 1
                else:
                    matched[(em,emotion)] += sentiment['sentiment_score']
    for em, tot in occurences.items():
        pos = unmatched[(em, 'Positive')]
        neg = unmatched[(em, 'Negative')]
        neu = tot - (pos +neg)
        assert(neu>=0)
        unmatchedFile.write(",".join((em,'-',str(tot),'1',str(neg),str(neu),str(pos),'-','Unknown')) + '\n')
    matchedFile.writelines("\n".join([elem[0] + ',' + elem[1] + ',' + str(cnt) for elem, cnt in matched.items()]))
    #unmatchedFile.writelines("\n".join([elem[0] + ',' + elem[1] + ',' + str(cnt) for elem, cnt in unmatched.items()]))
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