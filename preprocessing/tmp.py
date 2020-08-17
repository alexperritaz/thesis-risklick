# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 04:14:22 2020

@author: Alex
"""

# import torch
# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# sentence = 'A procedure used to transform data. Embeddings are fascinating.'

# tokenized_text = tokenizer.tokenize(sentence)
# print(tokenized_text)


import pandas as pd
import numpy as np

df = pd.read_csv('./../../data/interim/tags_few/sampled_all_per_class/cleaned.csv')
df_labels = pd.read_csv('./../../data/common/risks/risks_encoded.csv')
# Get empty counts
n_classes = df_labels['classes'].nunique()


for i in range(len(df)):
    print(i)

    
    
    
# for column in df.columns:
#     print(column)
#     int_null_count = len(df[df[column] == -1])
#     str_null_count = len(df[df[column] == '-1'])
    
#     total_null_count = int_null_count + str_null_count
#     print(total_null_count)
    
#     pass