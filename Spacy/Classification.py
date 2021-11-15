import re
import string
import pandas as pd
import numpy as np

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df_amazon = pd.read_excel(r'\amazon_alexa.xlsx').head(500)
X         = df_amazon['verified_reviews'] 
ylabels   = df_amazon['feedback']

def spacy_tokenizer(sentence):
    
    # configuration
    mytokens = []
    punctuations = string.punctuation
    nlp = spacy.load("en_core_web_sm")
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    
    # Remove trailling and overflow white spaces
    sentence = re.sub("\s\s+" , " ", sentence.strip()) #takes too much time
    
    # Lemmatizing each token and converting each token into lowercase
    mytokens = [mytokens.append(word.lemma_) or word.lemma_ for word in nlp(sentence)] # it automatically lowercase the letters

    # Removing stop words and punctuation
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens

### Test
#%%time
#sentences = sentences.tolist()
#df_tmp = list(map(lambda sentence : spacy_tokenizer(sentence), sentences))
#df_tmp[:3]

%%time
count_vect = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))  # stop_words=german_stop_words # , ngram_range=(1, 1)
X = count_vect.fit_transform(X)

%%time
tfidf_transformer = TfidfTransformer(use_idf=True)
X = tfidf_transformer.fit_transform(X)

%%time
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.33)
model = LogisticRegression().fit(X_train, y_train)

%%time
# Predicting with a test dataset
predicted = model.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy : {:0.4f}".format(metrics.accuracy_score(y_test, predicted)))
print("Logistic Regression Precision: {:0.4f}".format(metrics.precision_score(y_test, predicted)))
print("Logistic Regression Recall   : {:0.4f}".format(metrics.recall_score(y_test, predicted)))

test_phrase = 'Fur den Inhalt der Mitteilung ist der Emittent Verantwortlich.'
