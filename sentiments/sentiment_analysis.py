import pandas as pd
import numpy as np

import string

from nltk import *
from nltk.corpus import stopwords


# USER COMMENTS SENTIMENT ANALYSIS

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def sentiment_analysis(comment):
    sentences_scores = {}

    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(comment)
    print('Score:', score)
    if score['compound'] >= 0.05:
        print(comment)
        #sentences_scores[key] = [comment, score['compound']]
        print("Positive")
        return 1

    elif score['compound'] <= - 0.05:
        print(comment)
        #sentences_scores[key] = [comment, score['compound']]
        print("Negative")
        return -1

    else:
        print(comment)
        #sentences_scores[key] = [comment, score['compound']]
        print("Neutral")
        return 0

sentiment_analysis('I am happy but today my mom died')



from nltk.corpus import movie_reviews as mr
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk

from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def sentiment_analysis2(text):
    test = []
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # text = [word for word in text.split(" ")]
    print('Text:', text)
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    print(pos_tags)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0],get_wordnet_pos(t[1])) if get_wordnet_pos(t[1]) else t[0] for t in pos_tags]

    # remove words with only one letter
    text = [t for t in text if len(t) > 1]

    # join all
    text = " ".join(text)
    print(text)
    mr_positive = 0
    mr_negative = 0
    for word, tag in pos_tags:
        synset = lesk(text, word, tag)
        print(synset)
        if synset is not None:
            mr_positive += swn.senti_synset(synset.name()).pos_score()
            print(mr_positive)

            mr_negative += swn.senti_synset(synset.name()).neg_score()
            print(mr_negative)
    print(mr_positive)
    print(mr_negative)
    test.append('pos' if mr_positive > mr_negative else 'neg')
    print(test)

    return test


