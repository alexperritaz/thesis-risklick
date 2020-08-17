# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 01:16:22 2020

@author: Alex
"""


import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import classification_report, confusion_matrix
# df = pd.read_csv('../../data/processed/tags_few/embedded/embedded_bert_avg.csv')

# print(df.isnull().values.sum())

# # df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

# df = df.fillna(0)

# print(df.isnull().values.sum())

# df.to_csv('../../data/processed/tags_few/embedded/embedded_bert_avg_corrected.csv')

# model = 'CNN'
# storage_path = 'sqlite:///../models/tags_few_20000/bert_mean_encoded_2/%s/study_optuna.db' % (model)
# csv_path = '../models/tags_few_20000/bert_mean_encoded_2/%s/study_params.csv' % (model)
# study = optuna.create_study(direction='maximize', study_name=model, storage=storage_path, load_if_exists=True)
# study.trials_dataframe().to_csv(csv_path, index=False, header=False)


df = pd.read_csv('glove_preds.csv')

y_true = df['TRUE']






y_pred = df['RF']
print(confusion_matrix(y_pred,y_true))        
print(classification_report(y_pred,y_true))
y_pred = df['LogReg']
print(confusion_matrix(y_pred,y_true))        
print(classification_report(y_pred,y_true))
y_pred = df['MLP']
print(confusion_matrix(y_pred,y_true))        
print(classification_report(y_pred,y_true))
y_pred = df['CNN']
print(confusion_matrix(y_pred,y_true))        
print(classification_report(y_pred,y_true))
y_pred = df['RNN']
print(confusion_matrix(y_pred,y_true))        
print(classification_report(y_pred,y_true))
y_pred = df['LSTM']
print(confusion_matrix(y_pred,y_true))        
print(classification_report(y_pred,y_true))
y_pred = df['GRU']
print(confusion_matrix(y_pred,y_true))        
print(classification_report(y_pred,y_true))