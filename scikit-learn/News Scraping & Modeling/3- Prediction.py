### Libs

import pandas as pd
import numpy as np
from functools import lru_cache

import spacy

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier

from matplotlib import pyplot

from joblib import dump, load

### Data

df = pd.read_pickle(r"\news_new83.pkl")
print(len(df))
# df[0]
# df[0]['url']

### Classifier

clf = load(r'\model_v0.joblib')

### Count_Vect

# Load the Count Vectorizer
count_vect = load(r'\countvect_v0.joblib')

### TFIDF Transformer

# Load the TFIDF Transformer
tfidf_transfomer = load(r'\tfidftransformer_v0.joblib')

### Change Dictionnaire into DataFrame

%%time

urls   = []
titles = []
texts  = []

for row in range(len(df)):
    urls.append(df[row]['url'])
    titles.append(df[row]['title'])
    texts.append(df[row]['text'])
    
df = pd.DataFrame(columns=['Url', 'Title', 'Text'])
df['Url']   = urls
df['Title'] = titles
df['Text']  =  texts
df.head()

### Process Data

%%time

texts = count_vect.transform(texts)

%%time

texts = tfidf_transfomer.transform(texts)

### Prediction

%%time
predicted = clf.predict(texts)

### Export

%%time

df['Relevancy'] = predicted
df.to_excel(r'\New_Data_Classification.xlsx', index=False)
