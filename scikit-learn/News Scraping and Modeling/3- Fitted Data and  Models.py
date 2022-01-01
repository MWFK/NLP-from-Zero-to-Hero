### Libs

import re, os
import string
import pandas as pd
import numpy as np
from numpy import std
from numpy import mean
from functools import lru_cache

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
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot

### Data

news = pd.read_excel()

***

news = news.sample(frac=1, axis=1).sample(frac=1).reset_index(drop=True)

print(news['usefull'].value_counts())
print('Data shape: ', news.shape)
news.head()

### NLP Processing

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

### Model Tuned

# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    #level0.append(('lr'  , LogisticRegression()))
    #level0.append(('knn' , KNeighborsClassifier()))
    #level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('mlp', MLPClassifier(alpha=0.01, hidden_layer_sizes=11, max_iter=500, random_state=2))) # MLPClassifier(alpha=0.01, hidden_layer_sizes=11, max_iter=500, random_state=2)
    level0.append(('svm', SVC(C=0.1, gamma=1, kernel='linear')))           # SVM works relatively well when there is a clear margin of separation between classes. SVM is more effective in high dimensional spaces. SVM is effective in cases where the number of dimensions is greater than the number of samples. SVM is relatively memory efficient.
    #level0.append(('bayes', GaussianNB()))
    
    # define meta learner model
    level1 = LogisticRegression(C=0.5, class_weight='balanced', max_iter=3000, solver='saga')         # LogisticRegression(C=0.5, class_weight='balanced', max_iter=3000, solver='saga')
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    return model

# get a list of models to evaluate
def get_models():
    models             = dict()
    
    # LogisticRegression(C=0.5, class_weight='balanced', max_iter=3000, solver='saga')
    #models['lr']       = LogisticRegression() 
    
    #models['knn']     = KNeighborsClassifier()
    #models['cart']    = DecisionTreeClassifier()
    
    # MLPClassifier(alpha=0.01, hidden_layer_sizes=11, max_iter=500, random_state=2)
    #models['mlp']      = MLPClassifier() # can be applied to complex non-linear problems. Works well with large input data. Provides quick predictions after training. The same accuracy ratio can be achieved even with smaller data.
    
    # SVC(C=0.1, gamma=1, kernel='linear')
    #models['svm']      = SVC() # SVM works relatively well when there is a clear margin of separation between classes. SVM is more effective in high dimensional spaces. SVM is effective in cases where the number of dimensions is greater than the number of samples. SVM is relatively memory efficient.
    
    #models['bayes']   = GaussianNB()
    models['stacking'] = get_stacking()
    return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv      = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=71)
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    scores  = cross_validate(model, X, y, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise')
    return scores
    
 %%time

X         = news['text'].astype(str) 
ylabels   = news['usefull'].astype(int) 
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.25, shuffle=True, stratify=ylabels)

count_vect = CountVectorizer(tokenizer=spacy_tokenizer, encoding='utf-8', decode_error='ignore', strip_accents='unicode', lowercase=True, analyzer='word', ngram_range=(2, 2), max_features=55000)# tokenizer = spacy_tokenizer  
count_vect = count_vect.fit(X_train)
X_train    = count_vect.transform(X_train)

tfidf_transformer = TfidfTransformer(use_idf=True)
tfidf_transformer = tfidf_transformer.fit(X_train)
X_train           = tfidf_transformer.transform(X_train)

X_test = count_vect.transform(X_test)
X_test = tfidf_transformer.transform(X_test)

%%time

model = get_stacking()
model = model.fit(X_train, y_train)

predicted = model.predict(X_test)
print("Accuracy : {:0.4f}".format(metrics.accuracy_score (y_test, predicted)))
print("Precision: {:0.4f}".format(metrics.precision_score(y_test, predicted)))
print("Recall   : {:0.4f}".format(metrics.recall_score   (y_test, predicted)))

%%time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, "o-")
    axes[2].fill_between(
        fit_times_mean,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt




fig, axes = plt.subplots(3, 2, figsize=(10, 15))

title = "Learning Curves of the Stacked model"
plot_learning_curve(model, title, X_train, y_train, axes=axes[:, 0], ylim=(0.7, 1.01), n_jobs=-1)

plt.show()

