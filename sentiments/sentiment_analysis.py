import pandas as pd
import numpy as np

import string

from nltk import *
from nltk.corpus import stopwords


# USER COMMENTS SENTIMENT ANALYSIS

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
def sentiment_analysis(clear_sentences_dict):
    sentences_scores = {}
    for key, comment in clear_sentences_dict.items():
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(comment)
        print('Score:', score)
        if score['compound'] >= 0.05:
            print(comment)
            sentences_scores[key] = [comment, score['compound']]
            print("Positive")

        elif score['compound'] <= - 0.05:
            print(comment)
            sentences_scores[key] = [comment, score['compound']]
            print("Negative")

        else:
            print(comment)
            sentences_scores[key] = [comment, score['compound']]
            print("Neutral")

    is_bad_review = []
    compund = []
    df = pd.DataFrame.from_dict(clear_sentences_dict, orient='index', columns=['comment'])

    for key, comment in clear_sentences_dict.items():
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(comment)

        if score['pos'] == score['neg']:
            is_bad_review.append(0)
            compund.append(score['compound'])

        elif score['pos'] > score['neg']:
            is_bad_review.append(-1)
            compund.append(score['compound'])

        else:
            is_bad_review.append(1)
            compund.append(score['compound'])

    df['is_bad_review'] = is_bad_review
    df['compund'] = compund

    import seaborn as sns
    import matplotlib.pyplot as plt

    for x in [-1, 0, 1]:
        subset = df[df['is_bad_review'] == x]

        # Draw the density plot
        if x == -1:
            label = "Good reviews"

        elif x == 0:
            label = "Neutral"

        else:
            label = "Bad reviews"

        ax = plt.gca()
        subset = np.array(subset['compund']).astype(np.float)

        sns.distplot(subset, ax=ax, hist=False, label=label)
        ax.set_xlabel('compound score')
        ax.set_ylabel('univariate distribution of observations')

    plt.show()


from nltk.corpus import movie_reviews as mr
from nltk import pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk


def sentiment_analysis2(names):
    test = []
    for i in range(len(names)):
        if isinstance(comments[i], str):

            # text = tk(comments[i])
            # lower text
            text = comments[i].lower()

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
            # lemmatize text
            text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
            # remove words with only one letter
            text = [t for t in text if len(t) > 1]
            [all_words.append(word) for word in text]
            # join all
            text = " ".join(text)
            print(text)
            mr_positive = 0
            mr_negative = 0
            for word, tag in pos_tags:
                synset = lesk(text, word, tag)
                if synset is not None:
                    mr_positive += swn.senti_synset(synset.name()).pos_score()
                    print(mr_positive)

                    mr_negative += swn.senti_synset(synset.name()).neg_score()
                    print(mr_negative)

            test.append('pos' if mr_positive > mr_negative else 'neg')
    print(test)

    return test

