import sys
# Appeding our src directory to sys path so that we can import modules.
sys.path.append('../..') 

import src.tn.lib.Singleton
from  src.tn.lib.sentimoji import get_emoji_sentiment_rank

if __name__ == "__main__":
    print ("Hello world")
    print(get_emoji_sentiment_rank("😂"))