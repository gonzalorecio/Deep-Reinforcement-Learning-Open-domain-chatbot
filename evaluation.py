from collections import Counter
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# doc_1 = 'Convolutional Neural Networks are very similar to ordinary Neural Networks from the previous chapter'
# doc_2 = 'Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way.'
# doc_3 = 'In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth.'
# lista = [doc_1, doc_2, doc_3]

def diversity(docs):
    docs = (' '.join(filter(None, docs))).lower()
    tokens = word_tokenize(docs)
    tokens = [t for t in tokens if t not in stop_words]
    word_l = WordNetLemmatizer()
    tokens = [word_l.lemmatize(t) for t in tokens if t.isalpha()]

    uni_grams = list(set(tokens))
    bi_grams = list(ngrams(tokens, 2))

    counter_unigrams = uni_grams
    counter_bigrams  = Counter(bi_grams)
    return counter_unigrams, counter_bigrams





