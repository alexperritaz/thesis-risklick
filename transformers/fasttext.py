# -*- coding: utf-8 -*-

import re
import random
import numpy as np
import pandas as pd
import lxml.etree as ET
from utils import walk_files, progress_bar
from sklearn.model_selection import train_test_split



# def transform_XML(filepaths, labels, src_directory, dst_file):
#     df_labels = pd.read_csv(src_labels)
#     n_classes = df_labels['classes'].nunique()    
#     documents = []
#     fasttext = ''    
#     for i in range(n_classes):
#         classes_i = df_labels.loc[df_labels['classes'] == i]
#         sorted_i = sorted(classes_i['nct_id'])
#         for path in progress_bar(filepaths, "Extracting class %s : " % (i)):
#             if any(nct_id in path for nct_id in sorted_i):
#                 xmlstr = ET.tostring(ET.parse(path), encoding='utf-8', method='xml').decode('utf-8')
#                 xmlstr = re.sub(' +', ' ', xmlstr)
#                 xmlstr = xmlstr.replace('\n', '')
#                 documents.append(fasttext + "__label__" + str(i) + " " + xmlstr + "\n")
#         pass
    
#     # np.random.shuffle(documents)
#     text_documents = ''.join(documents)
#     with open(dst_file, "w", encoding='utf-8-sig') as txt_file:
#         txt_file.write(text_documents)
#     pass

def transform_XML(filepaths, labels, src_directory, dst_file):
    print(type(filepaths))
    documents = []
    fasttext = ''
    np.savetxt("transformers_labels.csv", labels, delimiter=",")
    for index,label in progress_bar(enumerate(labels), "Extracting :"):
        path = filepaths.iloc[index]
        xmlstr = ET.tostring(ET.parse(path.filename), encoding='utf-8', method='xml').decode('utf-8')
        xmlstr = re.sub(' +', ' ', xmlstr)
        xmlstr = xmlstr.replace('\n', '')
        documents.append(fasttext + "__label__" + str(int(label[0])) + " " + xmlstr + "\n")
        pass
    text_documents = ''.join(documents)
    with open(dst_file, "w", encoding='utf-8-sig') as txt_file:
        txt_file.write(text_documents)
    
    
    pass

# random.seed(32)
np.random.seed(69)
src_directory = '../../data/interim/tags_few/sampled_20000_per_class/xml/'
src_labels = '../../data/interim/tags_few/sampled_20000_per_class/labels_encoded.csv'
filepaths = walk_files(src_directory)
# np.random.shuffle(filepaths)

data_x = pd.DataFrame(filepaths, columns=['filename']) 
df_labels = pd.read_csv(src_labels)
data_y = np.array(df_labels.to_numpy()[:,1:], dtype=np.float)

X_trainval, X_test, y_trainval, y_test = train_test_split(data_x, data_y, test_size=0.2, stratify=data_y, random_state=69)

# print(X_test[0:20])
# print(y_test[0:20])
# splits = [0.6, 0.8]
# train, test = np.split(filepaths, [int(0.8*len(filepaths))])
# y_train, y_test = np.split(labels, [int(0.8*len(labels))])
# train, dev, test = np.split(filepaths,[int(splits[0]*len(filepaths)), int(splits[1]*len(filepaths))])
# print(len(test))
# print(test[0])
# print(y_test[0:20])


# dst_file = '../../data/interim/tags_few/sampled_20000_per_class/fasttext_distr/fasttext_train.txt'
# transform_XML(train, src_directory, src_labels, dst_file)
# dst_file = '../../data/interim/tags_few/sampled_20000_per_class/fasttext_distr/fasttext_dev.txt'
# transform_XML(dev, src_directory, src_labels, dst_file)
dst_file = '../../data/interim/tags_few/sampled_20000_per_class/fasttext_distr/fasttext_test.txt'
transform_XML(X_test,y_test, src_directory, dst_file)
# transform_XML(test, src_directory, src_labels, dst_file)