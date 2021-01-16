from collections import Counter
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

doc_1 = 'Convolutional Neural Networks are very similar to ordinary Neural Networks from the previous chapter'
doc_2 = 'Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way.'
doc_3 = 'In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth.'
lista = [doc_1, doc_2, doc_3]

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


# ANOVA TESTING HYPOTHESIS: https://www.reneshbedre.com/blog/anova.html

# I am using Python 3
# load packages
import pandas as pd
# load data file
df = pd.read_csv("https://reneshbedre.github.io/assets/posts/anova/onewayanova.txt", sep="\t")
# reshape the d dataframe suitable for statsmodels package
df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['A', 'B', 'C', 'D'])
# replace column names
df_melt.columns = ['index', 'treatments', 'value']

# generate a boxplot to see the data distribution by treatments. Using boxplot, we can
# easily detect the differences between different treatments
import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.boxplot(x='treatments', y='value', data=df_melt, color='#99c2a2')
ax = sns.swarmplot(x="treatments", y="value", data=df_melt, color='#7d0013')
plt.show()


# load packages
import scipy.stats as stats
# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
fvalue, pvalue = stats.f_oneway(df['A'], df['B'], df['C'], df['D'])
print(fvalue, pvalue)
# 17.492810457516338 2.639241146210922e-05

# get ANOVA table as R like output
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Ordinary Least Squares (OLS) model
model = ols('value ~ C(treatments)', data=df_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)
# ANOVA table using bioinfokit v1.0.3 or later (it uses wrapper script for anova_lm)
from bioinfokit.analys import stat
res = stat()
res.anova_stat(df=df_melt, res_var='value', anova_model='value ~ C(treatments)')
print(res.anova_summary)