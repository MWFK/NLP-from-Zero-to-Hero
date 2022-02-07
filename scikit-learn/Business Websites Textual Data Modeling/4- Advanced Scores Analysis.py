### Objectives
'''Find the most impactfull key words based on the results of the previous notebook(all websites textual data analysis) for each web site.'''

### Libs
import pandas as pd
import operator
import nltk


### Data
df = pd.read_excel(r'***')
whiteList_de = ['***'...]
print(len(whiteList_de))
blackList_de = ['***'...]
print(len(blackList_de))
print(df.shape)
df.head()

### Scores WhiteList
texts = df['Text'].tolist()
companies = df['Company'].tolist()


# def Similarity(key_words, text):
#     return sum([text.count(word) for word in key_words])
def Similarity(key_words, text):
    num=0
    for word in key_words:
        if text.count(word)!=0:
            print(word, text.count(word))
            num+=text.count(word)
    return num


def scoring_similarity_w(whiteList_de, data):
    return list(map(lambda article: Similarity(whiteList_de, article), data))


occ_word      = []
occ_word_list = []
for text in texts:
    for word in whiteList_de:
        occ_word.append(text.count(word))
        
    occ_word_list.append(occ_word)
    occ_word = []


companies_words_w = []
for idx,company in enumerate(companies):
    companies_words_w.append([companies[idx], sorted(list(zip(whiteList_de, occ_word_list[idx])), key = operator.itemgetter(1))[-5:]])
score_1 = scoring_similarity_w(whiteList_de, texts)

print('Score_1 \n', score_1, '\n')
white_score = list(zip(score_1, companies_words_w))
print(white_score[0])
white_score[0][0]

tmp = df.loc[df['Company'] == 'Company name']
tmp


score_1 = scoring_similarity_w(whiteList_de, tmp['Text'])
print('Score_1 \n', score_1, '\n')


### Scores BlackList
occ_wordb      = []
occ_word_listb = []


for text in texts:
    for word in blackList_de:
        occ_wordb.append(text.count(word))
        
    occ_word_listb.append(occ_wordb)
    occ_wordb = []
Wall time: 504 ms
companies_words_b = []


for idx,company in enumerate(companies):
    companies_words_b.append([companies[idx], sorted(list(zip(blackList_de, occ_word_listb[idx])), key = operator.itemgetter(1))[-2:]])


def scoring_similarity_b(blackList_de, data):
    blackList_de_score = list(map(lambda article: Similarity(blackList_de, article), data))
    blackList_de_score = [i * (-2) for i in blackList_de_score]
    return blackList_de_score


score_1b = scoring_similarity_b(blackList_de, texts)
black_score = list(zip(score_1b, companies_words_b))
print(black_score[0])
black_score[0][0]


tmp = df.loc[df['Company'] == 'company name']
tmp


score_1 = scoring_similarity_b(blackList_de, tmp['Text'])
print('Score_1 \n', score_1, '\n')


### Sum of White & Black Scores
ws = [x[0] for x in white_score]
bs = [x[0] for x in black_score]
score = list(map(operator.add, ws,bs))
score[:20]

df_w = pd.DataFrame(companies_words_w, columns = ['Company_URL', 'WhiteList_Rank'])
df_b = pd.DataFrame(companies_words_b, columns = ['Company_URL', 'BlackList_Rank'])
df['WhiteList_Rank'] = df_w['WhiteList_Rank']
df['BlackList_Rank'] = df_b['BlackList_Rank']
df.columns

cols = ['***', ]

df = df[cols]
df.to_excel(r'***', index=False)
