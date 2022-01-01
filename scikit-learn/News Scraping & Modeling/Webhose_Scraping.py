import pandas as pd
import webhoseio
import os
PROXY = "***"
os.environ["HTTP_PROXY"]  = PROXY
os.environ["HTTPS_PROXY"] = PROXY

# search = {'language':'german', 
#           'thread.country':'DE', 
#           'site_type':'news', 
#           'thread.title':('übernimmt OR kauft OR übernahme OR übergibt OR veräuss* OR erwir* OR erwerb* OR kauft OR  verkauft OR schluck* OR kaufen OR beteilig* OR "hört auf" OR *beteiligung'),  
#           'text':('Mitarbeiter OR Arbeitsplätze OR Angestellte OR Mitarbeitende OR immobilien OR Verlust OR gewinn OR umsatz OR firma OR unternehmen OR sitz OR gruppe OR group OR Mittelstand OR mittelständisch'), 
#           'is_first':'true',
#           #'published':'>1564610400000',
#           }

# token   = '2bf759c0-3866-4ff6-a219-82badc4ad312'
# webhoseio.config(token=token)
# https://webhose.io/web-content-api
output = webhoseio.query('filterWebContent?token=***&format=json&ts=1629118130768&sort=crawled&q=language%3Agerman%20thread.country%3ADE%20site_type%3Anews%20thread.title%3A(%C3%BCbernimmt%20OR%20kauft%20OR%20%C3%BCbernahme%20OR%20%C3%BCbergibt%20OR%20ver%C3%A4uss*%20OR%20erwir*%20OR%20erwerb*%20OR%20kauft%20OR%20verkauft%20OR%20schluck*%20OR%20kaufen%20OR%20beteilig*%20OR%20%22h%C3%B6rt%20auf%22%20OR%20*beteiligung)%20text%3A(Mitarbeiter%20OR%20Arbeitspl%C3%A4tze%20OR%20Angestellte%20OR%20Mitarbeitende%20OR%20immobilien%20OR%20Verlust%20OR%20gewinn%20OR%20umsatz%20OR%20firma%20OR%20unternehmen%20OR%20sitz%20OR%20gruppe%20OR%20group%20OR%20Mittelstand%20OR%20mittelst%C3%A4ndisch)%20is_first%3Atrue%20published%3A%3E1564610400000')
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

