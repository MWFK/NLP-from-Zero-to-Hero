import pandas as pd
import webhoseio
import os
PROXY = "***"
os.environ["HTTP_PROXY"]  = PROXY
os.environ["HTTPS_PROXY"] = PROXY

# search = {'language':'german', 
#           'thread.country':'DE', 
#           'site_type':'news', 
#           'thread.title':(' *** OR ***'),  
#           'text':(' *** OR ***'), 
#           'is_first':'true',
#           #'published':'>**',
#           }

# token   = '**'
# webhoseio.config(token=token)
# https://webhose.io/web-content-api
output = webhoseio.query('filterWebContent?token=***&format=json&***')
print(output)

# while int(output['moreResultsAvailable'])>0:
#     for post in output['posts']:
#         print()
#     output = webhoseio.get_next()


print(int(output['totalResults']))
print(int(output['moreResultsAvailable']))

i = 0
title, date, text, url = [], [], [], []
while int(output['moreResultsAvailable'])> 0: # each request returns 20 titles
    for post in output['posts']:
        i+=1
        print(i)
        title.append(post['title'])
        date.append(post['published'])
        text.append(post['text'])
        url.append(post['thread']['site'])
    output = webhoseio.get_next()
    
df = pd.DataFrame(columns=['Title', 'Date', 'Site', 'Text'])
df['Title'] = title
df['Date'] = date
df['Site'] = url
df['Text'] = text
df.head()

