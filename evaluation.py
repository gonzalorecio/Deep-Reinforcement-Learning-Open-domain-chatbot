from collections import Counter
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

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


## COLLECT DATA :


import glob

RL_experiments, Baseline_experiments = [], []
old_group_experiments, young_group_experiments = [],[]

for file in glob.glob("results/*.csv"):
    if 'RL' in file:
        RL_experiments.append(file)
    else:
        Baseline_experiments.append(file)

    df = pd.read_csv(file)

    if 'RL' in file:
        if df['age'].values[0] >= 55:
            old_group_experiments.append(file)
        else:
            young_group_experiments.append(file)

print(Baseline_experiments)
print(RL_experiments)
print(old_group_experiments)
print(young_group_experiments)



def attribute(att,list_path):
    all_att_values = []
    for path in list_path:
        df = pd.read_csv(path)
        att_value = df[att][0]
        all_att_values.append(att_value)
    return all_att_values

def boxplot(path,name):
    import pandas as pd
    # load data file
    df = pd.read_csv(path, sep="\t")
    # reshape the d dataframe suitable for statsmodels package
    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['Young-range', 'Old-range'])
    # replace column names
    df_melt.columns = ['index', ' ', name]

    # generate a boxplot to see the data distribution by treatments. Using boxplot, we can
    # easily detect the differences between different treatments
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6,4))
    plt.title(name)
    my_pal = sns.color_palette("tab10")
    ax = sns.boxplot(x=' ', y=name, data=df_melt, palette=my_pal)
    ax = sns.swarmplot(x=" ", y=name, data=df_melt, color='#7d0013')
    ax.axes.set_title(name, fontsize=15)


    plt.savefig('evaluation_hypothesis/'+name+'.png')
    plt.show()

# Hypothesis 1
# print(attribute('dialogue_len',Baseline_experiments))
# baseline_dialogue_len_average = np.mean(attribute('dialogue_len',Baseline_experiments))
# RL_dialogue_len_average = np.mean(attribute('dialogue_len',RL_experiments))
# list1 = [attribute('dialogue_len',Baseline_experiments),attribute('dialogue_len',RL_experiments)]
# df = pd.DataFrame([])
# col1 = pd.Series(list1[0])
# col2 = pd.Series(list1[1])
# df['RL'] = col1.values
# df['Baseline'] = col2.values
# df.to_csv(r'evaluation_hypothesis/H1_dialogue_len.txt', header=['Baseline', 'RL'], index=None, sep='\t', mode='a')
#
#
# baseline_duration_average = np.mean(attribute('duration',Baseline_experiments))
# RL_duration_average = np.mean(attribute('duration',RL_experiments))
# list1 = [attribute('duration',Baseline_experiments),attribute('duration',RL_experiments)]
# df = pd.DataFrame([])
# col1 = pd.Series(list1[0])
# col2 = pd.Series(list1[1])
# df['RL'] = col1.values
# df['Baseline'] = col2.values
# df.to_csv(r'evaluation_hypothesis/H1_duration.txt', header=['Baseline', 'RL'], index=None, sep='\t', mode='a')
#
# baseline_uni_average = np.mean(attribute('len_uni',Baseline_experiments))
# RL_uni_average = np.mean(attribute('len_uni',RL_experiments))
# list1 = [attribute('len_uni',Baseline_experiments),attribute('len_uni',RL_experiments)]
# df = pd.DataFrame([])
# col1 = pd.Series(list1[0])
# col2 = pd.Series(list1[1])
# df['RL'] = col1.values
# df['Baseline'] = col2.values
# df.to_csv(r'evaluation_hypothesis/H1_len_uni.txt', header=['Baseline', 'RL'], index=None, sep='\t', mode='a')
#
#
# baseline_bigrams_average = np.mean(attribute('len_bigrams',Baseline_experiments))
# RL_bigrams_average = np.mean(attribute('len_bigrams',RL_experiments))
# list1 = [attribute('len_bigrams',Baseline_experiments),attribute('len_bigrams',RL_experiments)]
# df = pd.DataFrame([])
# col1 = pd.Series(list1[0])
# col2 = pd.Series(list1[1])
# df['RL'] = col1.values
# df['Baseline'] = col2.values
# print(df.head())
#df.to_csv(r'evaluation_hypothesis/H1_len_bigrams.txt', header=['Baseline', 'RL'], index=None, sep='\t', mode='a')



# # Hypothesis 2
# young_dialogue_len_average = np.mean(attribute('dialogue_len',young_group_experiments))
# old_dialogue_len_average = np.mean(attribute('dialogue_len',old_group_experiments))
# list1 = [attribute('dialogue_len',young_group_experiments),attribute('dialogue_len',old_group_experiments)]
# df = pd.DataFrame([])
# col1 = pd.Series(list1[0])
# col2 = pd.Series(list1[1]+[0,0])
# df['Young-range'] = col1.values
# df['Old-range'] = col2.values
# df.to_csv(r'evaluation_hypothesis/H2_dialogue_len.txt', header=['Young-range', 'Old-range'], index=None, sep='\t', mode='a')
#
#
#
#
# young_duration_average = np.mean(attribute('duration',young_group_experiments))
# old_duration_average = np.mean(attribute('duration',old_group_experiments))
# list1 = [attribute('duration',young_group_experiments),attribute('duration',old_group_experiments)]
# df = pd.DataFrame([])
# col1 = pd.Series(list1[0])
# col2 = pd.Series(list1[1]+[0,0])
# df['Young-range'] = col1.values
# df['Old-range'] = col2.values
# df.to_csv(r'evaluation_hypothesis/H2_duration.txt', header=['Young-range', 'Old-range'], index=None, sep='\t', mode='a')
#
# young_uni_average = np.mean(attribute('len_uni',young_group_experiments))
# old_uni_average = np.mean(attribute('len_uni',old_group_experiments))
# list1 = [attribute('len_uni',young_group_experiments),attribute('len_uni',old_group_experiments)]
# df = pd.DataFrame([])
# col1 = pd.Series(list1[0])
# col2 = pd.Series(list1[1]+[0,0])
# df['Young-range'] = col1.values
# df['Old-range'] = col2.values
# df.to_csv(r'evaluation_hypothesis/H2_len_uni.txt', header=['Young-range', 'Old-range'], index=None, sep='\t', mode='a')
#
#
# young_bigrams_average = np.mean(attribute('len_bigrams',young_group_experiments))
# old_bigrams_average = np.mean(attribute('len_bigrams',old_group_experiments))
# list1 = [attribute('len_bigrams',young_group_experiments),attribute('len_bigrams',old_group_experiments)]
# df = pd.DataFrame([])
# col1 = pd.Series(list1[0])
# col2 = pd.Series(list1[1]+[0,0])
# df['Young-range'] = col1.values
# df['Old-range'] = col2.values
# df.to_csv(r'evaluation_hypothesis/H2_len_bigrams.txt', header=['Young-range', 'Old-range'], index=None, sep='\t', mode='a')
#
#
# boxplot('evaluation_hypothesis/H2_dialogue_len.txt','H2_dialogue_len')
# boxplot('evaluation_hypothesis/H2_duration.txt','H2_duration')
# boxplot('evaluation_hypothesis/H2_len_uni.txt','H2_len_uni')
# boxplot('evaluation_hypothesis/H2_len_uni.txt','H2_len_bigrams')
#
#
# #Young_dialogue_len_average = np.mean(attribute('dialogue_len',young_group_experiments))
# #Old_dialogue_len_average = np.mean(attribute('dialogue_len',old_group_experiments))
#
# print(baseline_dialogue_len_average,RL_dialogue_len_average)
# print(baseline_duration_average,RL_duration_average)
# print(baseline_uni_average,RL_uni_average)
# print(baseline_bigrams_average,RL_bigrams_average)

#print(Young_dialogue_len_average,Old_dialogue_len_average)



def create_txt(attribute,list1,list2,H):
    df = pd.DataFrame([])

    #RL_dialogue_len = attribute(attribute,RL_experiments)
    #Base_dialogue_len = attribute(attribute,Baseline_experiments)
    list1_col = attribute(attribute,list1)
    list2_col = attribute(attribute,list2)

    col1 = pd.Series(list1_col)
    col2 = pd.Series(list2_col)
    if H == 1:
        df['RL'] = col1.values
        df['Baseline'] = col2.values
        df.to_csv(r'H' + str(H) + '_' + attribute +'.txt', header=['RL', 'Baseline'], index=None, sep='\t', mode='a')
    else:
        df['20-30'] = col1.values
        df['>=50'] = col2.values
        df.to_csv(r'H' + str(H) + '_' + attribute +'.txt', header=['20-30', '>=50'], index=None, sep='\t', mode='a')

def age_calculator(lista):
    ages = []
    for i in lista:
        year,month,day = i.split('-')
        age = 2021 - int(year) - ((int(month), int(day)) < (int(month), int(day))) - 1
        ages.append(age)
    return age





def likert_value(val):
    if val == 'Totalmente en desacurerdo':
        return 1
    elif val == 'En desacuerdo':
        return 2
    elif val == 'Ni de acuerdo ni en desacuerdo':
        return 3
    elif val == 'De acuerdo':
        return 4
    else:
        return 5

def create_col(df,list,name):
    col = pd.Series(list)
    df[name] = col.values

def questions(path='questionnaire/Old RL Spanish- Dialogue Generation Chatbot .csv'):

    df = pd.read_csv(path)
    df2 = pd.DataFrame([])

    # Ages
    birth_dates = list(df['Fecha de nacimiento'].values)
    ages = age_calculator(birth_dates)
    #create_col(df2,ages,'ages')

    # Speed
    speed = list(df['"El chatbot te respondió rápidamente" '].values)
    speed = [likert_value(element) for element in speed ]
    create_col(df2,speed,'speed')

    # Comprehension
    comprehension = list(df['"El chatbot te ha entendido bien" '].values)
    comprehension = [likert_value(element) for element in comprehension ]
    create_col(df2, comprehension, 'comprehension')

    # Understanding
    understanding = list(df['"Entendí lo que el chatbot me decía" '].values)
    understanding = [likert_value(element) for element in understanding]
    create_col(df2, understanding, 'understanding')

    # Kindness
    kindness = list(df['"El chatbot era amable"  '].values)
    kindness = [likert_value(element) for element in kindness]
    create_col(df2, kindness, 'kindness')

    # Frustration
    frustration = list(df['"La interacción del chatbot fue frustrante"'].values)
    frustration = [likert_value(element) for element in frustration]
    create_col(df2, frustration, 'frustration')

    # Friendly interface
    interface = list(df['"La interfaz del chatbot era agradable y bonita"'].values)
    interface = [likert_value(element) for element in interface]
    create_col(df2, interface, 'interface')

    # easy to use
    easy = list(df['"El chatbot era fácil de usar"'].values)
    easy = [likert_value(element) for element in easy]
    create_col(df2, easy, 'easy')

    # fun to use
    fun = list(df['"Fue divertido usar esa tecnología"'].values)
    fun = [likert_value(element) for element in fun]
    create_col(df2, fun, 'fun')

    # future use
    future = list(df['"Me gustaría usar este tipo de tecnología en el futuro"'].values)
    future = [likert_value(element) for element in future]
    create_col(df2, future, 'future')


    df2.to_csv(r'questionnaire/google_forms_RL_Old.txt', header=['speed', 'comprhension','understanding','kindness','frustration','interface','easy','fun','future'], index=None, sep='\t', mode='a')



#questions()







# ANOVA TESTING HYPOTHESIS: https://www.reneshbedre.com/blog/anova.html


# load packages
import scipy.stats as stats
from scipy.stats import ttest_ind

df1 = pd.read_csv('/Users/jreventos/Desktop/MAI/Semester 3/CIR/MAI-CIR/questionnaire/google_forms_RL_Young.txt',sep="\t")
df2 = pd.read_csv('/Users/jreventos/Desktop/MAI/Semester 3/CIR/MAI-CIR/questionnaire/google_forms_RL_Old.txt',sep="\t")

# stats f_oneway functions takes the groups as input and returns ANOVA F and p value
# fvalue, pvalue = stats.f_oneway(df1['speed'],df2['speed'])
# #fvalue, pvalue = stats.ttest_ind(df1['speed'],df2['speed'])
# print('Speed:', pvalue*100)
# fvalue, pvalue = stats.f_oneway(df1['comprhension'],df2['comprhension'])
# print('comprhension:',pvalue*100)
# fvalue, pvalue = stats.f_oneway(df1['understanding'],df2['understanding'])
# print('understanding:', pvalue*100)
# fvalue, pvalue = stats.f_oneway(df1['kindness'],df2['kindness'])
# print('kindness:', pvalue*100)
# fvalue, pvalue = stats.f_oneway(df1['frustration'],df2['frustration'])
# print('frustration:',pvalue*100)
# fvalue, pvalue = stats.f_oneway(df1['interface'],df2['interface'])
# print('interface:', pvalue*100)
# fvalue, pvalue = stats.f_oneway(df1['easy'],df2['easy'])
# print('easy:', pvalue*100)
# fvalue, pvalue = stats.f_oneway(df1['fun'],df2['fun'])
# print('fun:', pvalue*100)
# fvalue, pvalue = stats.f_oneway(df1['future'],df2['future'])
# print('future:', pvalue*100)

# Hypothesis 1:
df = pd.read_csv('/Users/jreventos/Desktop/MAI/Semester 3/CIR/MAI-CIR/evaluation_hypothesis/H2_dialogue_len.txt',sep="\t")
print(df.columns)
fvalue, pvalue = stats.f_oneway(df['Young-range'],df['Old-range'])
print('Dialogue Len:', pvalue*100)

df = pd.read_csv('/Users/jreventos/Desktop/MAI/Semester 3/CIR/MAI-CIR/evaluation_hypothesis/H2_duration.txt',sep="\t")
fvalue, pvalue = stats.f_oneway(df['Young-range'],df['Old-range'])
print('Duration:', pvalue*100)

df = pd.read_csv('/Users/jreventos/Desktop/MAI/Semester 3/CIR/MAI-CIR/evaluation_hypothesis/H2_len_uni.txt',sep="\t")
fvalue, pvalue = stats.f_oneway(df['Young-range'],df['Old-range'])
print('Len uni:', pvalue*100)

df = pd.read_csv('/Users/jreventos/Desktop/MAI/Semester 3/CIR/MAI-CIR/evaluation_hypothesis/H2_len_bigrams.txt',sep="\t")
fvalue, pvalue = stats.f_oneway(df['Young-range'],df['Old-range'])
print('Len bigrams:', pvalue)




# # get ANOVA table as R like output
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
#
# # Ordinary Least Squares (OLS) model
# model = ols('Likert-scale ~ C( )', data=df_melt).fit()
# anova_table = sm.stats.anova_lm(model, typ=9)
#
# print(anova_table)
# # ANOVA table using bioinfokit v1.0.3 or later (it uses wrapper script for anova_lm)
# from bioinfokit.analys import stat
# res = stat()
# res.anova_stat(df=df_melt, res_var='Likert-scale', anova_model='Likert-scale ~ C( )')
# print(res.anova_summary)



