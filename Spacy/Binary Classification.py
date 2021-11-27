import numpy as np
import pandas as pd
# from datetime import datetime

# !pip install spacy==3.1.1
# !pip install spacy-transformers
# pip install spacy-transformers --proxy *** --trusted-host pypi.org --trusted-host files.pythonhosted.org --default-timeout=10000
import spacy
import spacy_transformers
# Storing docs in binary format
from spacy.tokens import DocBin

# error !python -m spacy train /Code/config.cfg --output training/ --paths.train train.spacy --paths.dev dev.spacy --gpu-id -1
# solved here https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# https://newreleases.io/project/github/explosion/spacy-models/release/en_core_web_trf-3.0.0a0
# # Downloading the spaCy Transformer model "en_core_web_trf"
# !python -m spacy download en_core_web_trf

# # Install Spacy first 
# download https://newreleases.io/project/github/explosion/spacy-models/release/en_core_web_trf-3.0.0a0
# the put it in \Anaconda3\Lib\site-packages
# extract the .gz then the .tar (replace all files when prompt)
# then open cmd fom the folder that has the setup.py file within the extracted folders and run the following: python setup.py install
# the dictionnairy will be downloaded, but when in the cmd it shows that it's processing dependancies you need to see which one is making a problem then interupt the execution on the terminal with ctrl+c before it arrives to that dependancy (the problem changes from one env to another).
# restart the notebook kernal, and it works like a charm =)

nlp=spacy.load("en_core_web_sm") # version 3.0 does not work i should download version 3.2 # nlp=spacy.load("en_core_web_lg") ##nlp=spacy.load("en_core_web_trf")
nlp.pipe_names

df = pd.read_excel('amazon_alexa.xlsx')
print(df['verified_reviews'][2])
df=df[['verified_reviews' , 'feedback']]
print(df.shape)

train = df.sample(frac = 0.8, random_state = 25)
test = df.drop(train.index)
print(train.shape, test.shape)

%%time

records = train.to_records(index=False)
train = list(records)
print(train[0])

records_ = test.to_records(index=False)
test = list(records_)
print(test[0])

def document(data):

    text = []
    for doc, label in nlp.pipe(data, as_tuples = True):
        if (label==1):
            print("yes")
            doc.cats['positive'] = 1
            doc.cats['negative'] = 0
            doc.cats['neutral']  = 0
        elif(label==0):
            print("yess")
            doc.cats['positive'] = 0
            doc.cats['negative'] = 1
            doc.cats['neutral']  = 0
    text.append(doc)

    return(text)
  
 print(train[0][1])
a=document(train[0:2])
print(a)
a[0].cats

%%time

train_docs = document(train)
print(train_docs[0].cats)

#Creating binary document using DocBin function in spaCy
doc_bin = DocBin(docs = train_docs)

#Saving the binary document as train.spacy
doc_bin.to_disk("train.spacy")

%%time

test_docs = document(test)
doc_bin = DocBin(docs = test_docs)
doc_bin.to_disk("dev.spacy")

# https://spacy.io/usage/training#quickstart
# !python -m spacy init fill-config ./base_config1.cfg ./config.cfg
# I chnaged the directory since it gives me this error Error: Invalid value for 'BASE_PATH': File 'r'C:/Users/mayadi/Documents/Work/PWM/Code/New' does not exist.
!python -m spacy init fill-config /Code/base_config.cfg ./config.cfg
  
 %%time
!python -m spacy train Code/config.cfg --output training/ --paths.train train.spacy --paths.dev dev.spacy --gpu-id -1
  
 nlp= spacy.load(r'\training\model-best')

print(train[7])
print(train[7][0])

demo = nlp('i hate my job i hate alexa')
print('Result: ', 1 if demo.cats['positive'] > demo.cats['negative'] else 0)
demo.cats

