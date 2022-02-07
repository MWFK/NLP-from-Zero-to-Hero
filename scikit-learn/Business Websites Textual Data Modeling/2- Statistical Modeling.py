### Objectives

'''Clean the data and prepare it for modeling 
This version is only for german language'''


### Libs
import os
PROXY = "***" # IP:PORT or HOST:PORT
os.environ["HTTP_PROXY"]  = PROXY
os.environ["HTTPS_PROXY"] = PROXY
import re
import glob
import html
import math
import json
import string
import pickle
import operator
import numpy as np
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from langdetect import detect
import nltk
# nltk.set_proxy(None)
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from functools import lru_cache


### Data
df = pd.read_excel(r'***')
print(df.shape)
df.head()

### Black and White list in both German and English
whiteList_de = ['***'...]
blackList_de = ['***'...]

### Data Cleansing
def clean_companies_names(df):
    df = pd.Series(df)
    df = df.apply(lambda company: re.sub('http://', ''     ,     company))                       # remove http
    df = df.apply(lambda company: re.sub('https://', ''    ,     company))                       # remove https
    df = df.apply(lambda company: re.search(r'(https?://)?(www\.)?([^/]*)', company).group(0))   # keep base url
    df = df.apply(lambda company: 'www.'+company if not company.startswith('www.') else company) # add www. for some urls
    return df

'''We can Add German lemma
use spacy for deutsch language detection, or use the other lib in the previous notebook.
We can Add German Language Detection
use spacy for deutsch language detection, or use the other lib in the previous notebook.'''


def cleansing_text(df):
    
    string.punctuation    
    def __remove_punctuation(text):
        punctuationfree = "".join([i for i in text if i not in string.punctuation]) 
        return punctuationfree

    df = pd.Series(df)
    df = df.apply(lambda article: __remove_punctuation(article))       # remove punctuation
    df = df.apply(lambda article: re.sub(' +', ' ' , str(article)))    # delete multiple spaces
    df = df.apply(lambda article: str(article).lower())                # lower case all texts

    return df


### StopWords
def remove_stopwords(text):
    return ' '.join([word for word in word_tokenize(text) if not word in stopwords.words('german')]) # much faster than seperating tokenizaton from removing the stop words

### Similarity
# Look for String and Substring
#@lru_cache(maxsize=10000)
def Similarity(key_words, text):
    return sum([text.count(word) for word in key_words])
### This function is good for testing
# def Similarity(key_words, text):
#     num=0
#     for word in key_words:
#         if text.count(word)!=0:
#             print(word, text.count(word))
#             num+=text.count(word)
#     return num


#@lru_cache(maxsize=10000)
def scoring_frequency_similarity(score_1, len_txt):
    return list(map(lambda x : x[0] / x[1], zip(score_1, len_txt))) 

###################################### Similarity Score
#######################################################
def scoring_similarity(whiteList_de, blackList_de, data):
    whiteList_de_score = list(map(lambda article: Similarity(whiteList_de, article), data))
    blackList_de_score = list(map(lambda article: Similarity(blackList_de, article), data))
    blackList_de_score = [i * (-2) for i in blackList_de_score]
    score = list(map(operator.add, whiteList_de_score, blackList_de_score))
    return score

### Model
########################################## Data Cleansing
#########################################################
df['Text']    = df['Text'].apply   (lambda    article: cleansing_text(article))
df['Text']    = df['Text'].apply   (lambda    article: remove_stopwords(article))
df['Company'] = df['Company'].apply(lambda company: clean_companies_names(company))


###################################### Similarity Score
#######################################################
score_1 = scoring_similarity(whiteList_de, blackList_de, df['Text'])
print('Score_1 \n', score_1, '\n')
score_2 = scoring_frequency_similarity(score_1, df['Text'].apply(lambda text: len(text)))
print('Score_2 \n', score_2)


### Assign the Results
df['Score_1']    = score_1
df['Score_2']    = [x * 1000 for x in score_2]
#df.sort_values(by=['Score_2'], ascending=False, inplace=True)
#df.reset_index(drop=True, inplace=True)
df.to_excel(r'***', index=False)
print(df.shape)
df.head()

### We have merged the results with data from other sources that we cannot disclose
