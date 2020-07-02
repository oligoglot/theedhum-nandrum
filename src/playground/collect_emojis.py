import linecache
import sys
import emoji
import csv
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../..') 

from  src.tn.lib.sentimoji import get_emoji_sentiment_rank

def extract_emojis(s):
    return [c for c in s if c in emoji.UNICODE_EMOJI]

files = [
    '../../resources/data/tamil_dev.tsv',
    '../../resources/data/tamil_train.tsv',
    '../../resources/data/tamil_trial.tsv',
    '../../resources/data/malayalam_dev.tsv',
    '../../resources/data/malayalam_train.tsv',
    '../../resources/data/malayalam_trial.tsv',
]

matchedFn = "../../resources/data/matched_emojis.txt"
unmatchedFn = "../../resources/data/unmatched_emojis.txt"
matched = []
unmatched = []

with open(matchedFn, "w", encoding="UTF-8") as matchedFile, open(unmatchedFn, "w", encoding="UTF-8") as unmatchedFile:
    for fn in files:
        with open(fn, "r", encoding="utf-8") as mainFile:
            readTsv = csv.reader(mainFile, delimiter="\t")
            for row in readTsv:
                txt, emotion = row
                emojiji = extract_emojis(txt)
                for em in emojiji:
                    sentiment = get_emoji_sentiment_rank(em)
                    if sentiment == False:
                        if [em, emotion] not in unmatched:
                            unmatched.append([em, emotion])
                    else:
                        if [em, emotion] not in matched:
                            matched.append([em, emotion])

    matchedFile.writelines("\n".join([",".join(mat) for mat in matched]))
    unmatchedFile.writelines("\n".join([",".join(unmat) for unmat in unmatched]))