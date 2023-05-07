#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:44:30 2023

@author: namankhurpia
"""

import json
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from IPython.display import display

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Embedding
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split



#train dataset

with open('dataset/train.jsonl', 'r') as json_file:
    json_list = list(json_file)
    
    
output_list = []
for json_str in json_list:
    result = json.loads(json_str)
    extract1 = dict((key, value) for key, value in result.items() if key in ('uuid', 'postText', 'targetParagraphs', 'spoiler', 'tags', 'targetTitle', 'targetDescription', 'targetKeywords', 'provenance'))
    extract1['postText'] = " ".join(str(x) for x in extract1['postText'])
    extract1['targetParagraphs'] = " ".join(str(x) for x in extract1['targetParagraphs'])
    extract1['spoiler'] = " ".join(str(x) for x in extract1['spoiler'])
    extract1['tags'] = " ".join(str(x) for x in extract1['tags'])
    
    # Get the human spoiler from the provenance field
    if 'provenance' in result and 'humanSpoiler' in result['provenance']:
        extract1['human_spoiler'] = result['provenance']['humanSpoiler']
    else:
        extract1['human_spoiler'] = None
    
    output_list.append(extract1)
  
df = pd.json_normalize(output_list)


    
#tokenizing train dataset

def tokenize(text):
    return word_tokenize(text)

df['tokenized_postText'] = df['postText'].apply(tokenize)
df['tokenized_targetParagraphs'] = df['targetParagraphs'].apply(tokenize)
df['tokenized_spoiler'] = df['spoiler'].apply(tokenize)

#making csv

df['tags_nums'] = df['tags'].replace({"phrase": 0, "passage": 1, "multi": 2})

df.to_csv('train.csv', index=False)


##############################################################################

