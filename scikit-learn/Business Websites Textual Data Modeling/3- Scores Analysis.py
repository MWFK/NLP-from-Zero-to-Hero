### Objectives
'''Find the most impactfull key words based on the results of the previous notebook.'''

### Libs
import pandas as pd
import operator
import nltk


### Data
df = pd.read_excel(r'***')
print(df.shape)
df.head()

df['Text_Length'] = df['Text'].apply(lambda text: len(text))
df['Text_Length'].describe().to_excel(r'***', index=False)

whiteList_de = ['***'...]
print(len(whiteList_de))
blackList_de = ['***'...]
print(len(blackList_de))

### Join all texts and calculate the frequency of each word
texts = df['Clean_Text'].tolist()
print(len(texts))
text = ' '.join(texts)
print(len(text))
print(text[:1000])
# sentence_data = "The First sentence is about Python our . The Second: about Django. You can learn Python,Django and Data Ananlysis here. "

# ### False ###
# occurrences = [word for word in text.split()
#                     if word in 'touristik'] # using this we'll have occurence with is and our since they partially figure inside touristik
# print(len(occurrences))

# ### Correct ###
# occurrences = [word for word in sentence_data.split()
#                     if word in ['touristik']] # using this we'll have occurence only if word==touristik
# print(len(occurrences)) # 0

# ### Method 1
# %timeit print(sentence_data.count('touristik'))

# ### Method 2
# %timeit print(len([word for word in sentence_data.split() if word in set(['touristik', 'MK'])])) 

# ### Method 3
# %timeit print(len([word for word in sentence_data.split() if word in ['touristik','MK']])) # faster


occ_whiteList_de=[]
for word in whiteList_de:
    occ_whiteList_de.append(text.count(word))


print(sum(occ_whiteList_de))
list(zip(whiteList_de, occ_whiteList_de))
whiteList_de_scores = sorted(list(zip(whiteList_de, occ_whiteList_de)), key = operator.itemgetter(1))
print(type(whiteList_de_scores))


print(whiteList_de_scores[5][0])
print(whiteList_de_scores[5][1])


word  = []
score = []
for i in range(len(whiteList_de_scores)):
    word.append(whiteList_de_scores[i][0])
    score.append(whiteList_de_scores[i][1])

dfw = pd.DataFrame()
dfw['Word']  = word
dfw['score'] = score
dfw.to_excel(r'***')


occ_blackList_de=[]
for word in blackList_de:
    occ_blackList_de.append(text.count(word))


print(sum(occ_blackList_de))
list(zip(blackList_de, occ_blackList_de))
blackList_de_scores = sorted(list(zip(blackList_de, occ_blackList_de)), key = operator.itemgetter(1))


wordb  = []
scoreb = []
for i in range(len(blackList_de_scores)):
    wordb.append(blackList_de_scores[i][0])
    scoreb.append(blackList_de_scores[i][1])

dfb = pd.DataFrame()
dfb['Word']  = wordb
dfb['score'] = scoreb
dfb.to_excel(r'***')
