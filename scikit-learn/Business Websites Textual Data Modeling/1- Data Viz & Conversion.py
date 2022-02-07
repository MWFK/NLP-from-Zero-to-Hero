### Objectives
'''
1. Visualize the Data
2. Convert The data to a DataFrame
3. Clean the Data
'''

### Libs
import pandas as pd
import json


### Data Viz
data = pd.read_pickle(r'***')
# data.keys()
# data.values()
# data.items()
print(type(data))
print(len(list(data.keys())))

print(list(data.keys())[0],'\n\n')
print(data[list(data.keys())[0]]  [0],'\n\n')
print(data[list(data.keys())[0]]  [0]['title'],'\n\n')
print(len(data[  list(data.keys())[0]  ]),'\n\n')
print(list(data.keys())[:5])

# print(list(data.items())[1:2])
print(json.dumps(list(data.items())[1:2], indent=4))

print(data[list(data.keys())[1]] [1]['title'])
print(data[list(data.keys())[1]] [1]['text'])

web = list(data.keys())
print(web[:5])
web.index('website url')

data['website url'][:2]

### Convert Data to DataFrame
%%time

# Ietaret through keys, then get that key list, then iterate through the dicts inside the list
data_df = pd.DataFrame(columns=('Company', 'Text'))
for key in range(len(data.keys())): # The dictionnary keys
    text = ''
    for ldict in range(len(data[list(data.keys())[key]])): # the sub-dictionnary keys
        
        text = text + str(data[list(data.keys())[key]][ldict]['title']) + str(data[list(data.keys())[key]][ldict]['text'])
    
    data_df.loc[key] = [list(data.keys())[key], text]

data_df.to_excel(r'***', index=False) # Excel stores all the data but cannot show all of it !!!
df = data_df
print(df.shape)
df.head()
