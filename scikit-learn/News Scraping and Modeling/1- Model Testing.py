### Libs

import re, os
import string
import pandas as pd
import numpy as np
from functools import lru_cache
# accelerate
# https://stackoverflow.com/questions/62140599/countvectorizer-takes-too-long-to-fit-transform

import spacy
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate # to use sevrela validation metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics

### Data 

news = pd.read_excel()

***

X         = news['text'].astype(str) 
ylabels   = news['usefull']
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.25, shuffle=True, stratify=ylabels)

### NLP

punctuations = string.punctuation
nlp = spacy.load("de_core_news_sm")
stop_words = spacy.lang.de.stop_words.STOP_WORDS

@lru_cache(maxsize=10000)
def spacy_tokenizer(sentence):
    
    mytokens = []
    
#     # configuration
#     mytokens = []
#     punctuations = string.punctuation
#     nlp = spacy.load("de_core_news_sm")
#     stop_words = spacy.lang.de.stop_words.STOP_WORDS
    
#     # Remove trailling and overflow white spaces
#     sentence = re.sub("\s\s+" , " ", sentence.strip()) #takes too much time
    
    # Lemmatizing each token and converting each token into lowercase
    mytokens = [mytokens.append(word.lemma_) or word.lemma_ for word in nlp(re.sub("\s\s+" , " ", sentence.strip()))] # it automatically lowercase the letters

    # Removing stop words and punctuation
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens

### Test
#%%time
#sentences = sentences.tolist()
#df_tmp = list(map(lambda sentence : spacy_tokenizer(sentence), sentences))
#df_tmp[:3]

%%time

text_clf = Pipeline([('vect' , CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf'  , MultinomialNB())])

model     = text_clf.fit(X_train, y_train)
predicted = model.predict(X_test)

print("Logistic Regression Accuracy : {:0.4f}".format(metrics.accuracy_score (y_test, predicted)))
print("Logistic Regression Precision: {:0.4f}".format(metrics.precision_score(y_test, predicted)))
print("Logistic Regression Recall   : {:0.4f}".format(metrics.recall_score   (y_test, predicted)))

%%time

text_clf = Pipeline([('vect' , CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf'  , MultinomialNB())])

parameters = {'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
               'tfidf__use_idf'  : (True, False),
               'clf__alpha'      : (1e-2, 1e-3)}

gs_clf    = GridSearchCV(text_clf, parameters, n_jobs=-1)
model     = gs_clf.fit(X_train, y_train)
print(gs_clf.best_score_)
print(gs_clf.best_params_)

predicted = model.predict(X_test)
print("Accuracy : {:0.4f}".format(metrics.accuracy_score (y_test, predicted)))
print("Precision: {:0.4f}".format(metrics.precision_score(y_test, predicted)))
print("Recall   : {:0.4f}".format(metrics.recall_score   (y_test, predicted)))

%%time

text_clf = Pipeline([('vect' , CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf'  , MultinomialNB())])

parameters = {'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
               'tfidf__use_idf'  : (True, False),
               'clf__alpha'      : (1e-2, 1e-3)}

rs_clf    = RandomizedSearchCV(text_clf, parameters, n_jobs=-1)
model     = rs_clf.fit(X_train, y_train)
print(rs_clf.best_score_)
print(rs_clf.best_params_)

predicted = model.predict(X_test)
print("Accuracy : {:0.4f}".format(metrics.accuracy_score (y_test, predicted)))
print("Precision: {:0.4f}".format(metrics.precision_score(y_test, predicted)))
print("Recall   : {:0.4f}".format(metrics.recall_score   (y_test, predicted)))

%%time

X         = news['text'].astype(str) 
ylabels   = news['usefull']
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.25, shuffle=True, stratify=ylabels)

text_clf   = Pipeline([ ('vect'   , CountVectorizer()),
                        ('tfidf'  , TfidfTransformer()),                   
                        ('clf'    , SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))])

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],              
              'tfidf__use_idf'   : (True, False),
              'clf__alpha'       : (1e-2, 1e-3),
}

gs_clf   = GridSearchCV(text_clf, parameters, n_jobs=-1)
model_gs = gs_clf.fit(X_train, y_train)

print('{:0.4f}'.format(model_gs.best_score_))
print(model_gs.best_params_)

predicted = model.predict(X_test)
print("Accuracy : {:0.4f}".format(metrics.accuracy_score (y_test, predicted)))
print("Precision: {:0.4f}".format(metrics.precision_score(y_test, predicted)))
print("Recall   : {:0.4f}".format(metrics.recall_score   (y_test, predicted)))


### HP tuning of algos that are going to be used in the stacing model with meta learner
###Logistic Regression
%%time

news = news.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)

X         = news['text'].astype(str).head(50)
ylabels   = news['usefull'].astype(int).head(50)
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.25, shuffle=True, stratify=ylabels)

count_vect = CountVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', lowercase=True, analyzer='word', ngram_range=(2, 2), max_features=55000)# tokenizer = spacy_tokenizer  
count_vect = count_vect.fit(X_train)
X_train    = count_vect.transform(X_train)

tfidf_transformer = TfidfTransformer(use_idf=True)
tfidf_transformer = tfidf_transformer.fit(X_train)
X_train           = tfidf_transformer.transform(X_train)

X_test = count_vect.transform(X_test)
X_test = tfidf_transformer.transform(X_test)

param_grid = { 'solver'      : ['newton-cg', 'lbfgs','sag', 'saga'],
               'C'           : [0.5, 1, 1.5],
               'class_weight': [None, 'balanced'],
               'max_iter'    : [1000, 2000, 3000],
               'penalty'     : ['l2'] # l1 gives the following warning: One or more of the test scores are non-finite
             }
 
grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=3, n_jobs=-1)

# fitting the model for grid search
grid.fit(X_train, y_train)
# print best parameter after tuning
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

### SVC
%%time

news = news.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)

X         = news['text'].astype(str).head(50)
ylabels   = news['usefull'].astype(int).head(50)
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.25, shuffle=True, stratify=ylabels)

count_vect = CountVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', lowercase=True, analyzer='word', ngram_range=(2, 2), max_features=55000)# tokenizer = spacy_tokenizer  
count_vect = count_vect.fit(X_train)
X_train    = count_vect.transform(X_train)

tfidf_transformer = TfidfTransformer(use_idf=True)
tfidf_transformer = tfidf_transformer.fit(X_train)
X_train           = tfidf_transformer.transform(X_train)

X_test = count_vect.transform(X_test)
X_test = tfidf_transformer.transform(X_test)


param_grid = {'C'     : [0.1, 1, 10, 100, 1000],
              'gamma' : [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
 
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, n_jobs=-1)

# fitting the model for grid search
grid.fit(X_train, y_train)
# print best parameter after tuning
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

### MLP
%%time

news = news.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)

X         = news['text'].astype(str).head(50)
ylabels   = news['usefull'].astype(int).head(50)
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.25, shuffle=True, stratify=ylabels)

count_vect = CountVectorizer(encoding='utf-8', decode_error='ignore', strip_accents='unicode', lowercase=True, analyzer='word', ngram_range=(2, 2), max_features=55000)# tokenizer = spacy_tokenizer  
count_vect = count_vect.fit(X_train)
X_train    = count_vect.transform(X_train)

tfidf_transformer = TfidfTransformer(use_idf=True)
tfidf_transformer = tfidf_transformer.fit(X_train)
X_train           = tfidf_transformer.transform(X_train)

X_test = count_vect.transform(X_test)
X_test = tfidf_transformer.transform(X_test)

param_grid = {'solver': ['lbfgs', 'sgd', 'adam'], 
              'max_iter': [500, 1000, 1500], 
              'alpha': 10.0 ** -np.arange(1, 3), 
              'hidden_layer_sizes':np.arange(10, 12), 
              'random_state':[0, 1, 2]}
# learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’ ; ‘adaptive’ Only used when solver='sgd'.
grid = GridSearchCV(MLPClassifier(), param_grid, refit=True, verbose=3, n_jobs=-1)

# fitting the model for grid search
grid.fit(X_train, y_train)
# print best parameter after tuning
print(grid.best_params_)
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
