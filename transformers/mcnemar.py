# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 03:20:19 2020

@author: Alex
"""
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


df = pd.read_csv('mcnemar.csv')
print(df.columns)
truth = df['TRUE']
model_one = df['RF']
model_two = df['Transformers']



# t1 = truth == model_one
# t2 = truth == model_two

# t1_correct = sum(t1)
# t2_correct = sum(t2)
# A
# similarities = t1 == t2

# print(sum(truth == model_one))
# print(sum(truth == model_two))

# print(t1[0:10])
# print(t2[0:10])

# print(t1 == t2)


def get_frequencies(model_1, model_2) :
    # Case wrong by the model 1 and correct by the model 2
    wrong_1_correct_2 = np.count_nonzero(np.logical_and(model_1 == False, model_2))
    # Case correct by the model 1 and wrong by the model 2
    correct_1_wrong_2 = np.count_nonzero(np.logical_and(model_1, model_2 == False))
    # Case wrong by the model 1 and wrong by the model 2
    wong_1_wrong_2 = np.count_nonzero(np.logical_and(model_1 == False, model_2 == False))
    # Case correct by the model 1 and correct by the model 2
    correct_1_correct_2 = np.count_nonzero(np.logical_and(model_1, model_2))

    theoritical_frequencies = (wrong_1_correct_2 + correct_1_wrong_2) / 2

    p_value = (((wrong_1_correct_2 - theoritical_frequencies) ** 2) / theoritical_frequencies) + (((correct_1_wrong_2 - theoritical_frequencies) ** 2) / theoritical_frequencies)

    return  correct_1_correct_2, correct_1_wrong_2, wrong_1_correct_2,wong_1_wrong_2, p_value


correct_1_correct_2, correct_1_wrong_2, wrong_1_correct_2,wong_1_wrong_2, p_value = get_frequencies(model_one, model_two)

print(correct_1_correct_2)
print(correct_1_wrong_2)
print(wrong_1_correct_2)
print(wong_1_wrong_2)
print(p_value)

table = [[correct_1_correct_2,correct_1_wrong_2],
         [wrong_1_correct_2,wong_1_wrong_2]]

result = mcnemar(table, exact=True)

print('stats : %s, p-value : %s' %(result.statistic, result.pvalue))