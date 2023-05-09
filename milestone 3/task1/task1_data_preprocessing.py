# -*- coding: utf-8 -*-
"""task1 - data preprocessing"""

#train set
import json

with open('train.jsonl', 'r') as json_file:
    json_list = list(json_file)

output_list = []
for json_str in json_list:
    result = json.loads(json_str)
    extract1 = dict((key, value) for key, value in result.items() if key in ('uuid', 'postText', 'targetParagraphs', 'targetTitle', 'spoiler', 'tags'))
    extract1['postText'] = " ".join(str(x) for x in extract1['postText'])
    extract1['targetParagraphs'] = "\n".join(str(x) for x in extract1['targetParagraphs'])
    extract1['targetTitle'] = "".join(str(x) for x in extract1['targetTitle'])
    extract1['spoiler'] = " ".join(str(x) for x in extract1['spoiler'])
    extract1['tags'] = " ".join(str(x) for x in extract1['tags'])
    print(f"extract: {extract1}")
    output_list.append(extract1)

#save the train df
import json
import pandas as pd    

df = pd.json_normalize(output_list)
df.to_csv('output-1.csv', index=False)

#test set
import json
import pandas as pd    

with open('validation.jsonl', 'r') as json_file:
    json_list = list(json_file)


output_list_valid = []
for json_str in json_list:
    result = json.loads(json_str)
    extract1 = dict((key, value) for key, value in result.items() if key in ('uuid', 'postText', 'targetParagraphs', 'targetTitle', 'spoiler', 'tags'))
    extract1['postText'] = " ".join(str(x) for x in extract1['postText'])
    extract1['targetParagraphs'] = "\n".join(str(x) for x in extract1['targetParagraphs'])
    extract1['targetTitle'] = "".join(str(x) for x in extract1['targetTitle'])
    extract1['spoiler'] = " ".join(str(x) for x in extract1['spoiler'])
    extract1['tags'] = " ".join(str(x) for x in extract1['tags'])
    print(f"extract: {extract1}")
    output_list_valid.append(extract1)

df_valid = pd.json_normalize(output_list_valid)

#save the test df
import json
import pandas as pd    

df_valid = pd.json_normalize(output_list_valid)
df_valid.to_csv('valid_output.csv.csv', index=False)

"""#preprocessing"""

import pandas as pd

df = pd.read_csv('output-1.csv')
df['context'] = df['postText'] + ". " + df['targetParagraphs']
train_df = df[['context', 'tags']]
train_df['tags'] = train_df['tags'].replace({"phrase": 0, "passage": 1, "multi": 2})
train_df.columns=['text','label']

test_df = pd.read_csv('valid_output.csv')
test_df['context'] = test_df['postText'] + ". " + test_df['targetParagraphs']
test_df = test_df[['context', 'tags']]
test_df['tags'] = test_df['tags'].replace({"phrase": 0, "passage": 1, "multi": 2})
test_df.columns=['text','label']

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.2)

train_df.reset_index(drop=True).to_csv('df_train.csv',index=False)
val_df.reset_index(drop=True).to_csv('df_valid.csv',index=False)
test_df.reset_index(drop=True).to_csv('df_test.csv',index=False)
