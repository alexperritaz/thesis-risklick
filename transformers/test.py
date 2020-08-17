# -*- coding: utf-8 -*-


from flair.models import TextClassifier
from flair.data import Sentence
from sklearn.metrics import classification_report, confusion_matrix
from utils import progress_bar
import numpy as np

classifier = TextClassifier.load('../models/tags_few_20000/bert_mean_encoded_2/Transformers/50_epoch_last_4/best-model.pt')
# classifier = TextClassifier.load('../models/tags_few_20000/glove/Transformers/50_epoch_last_4/best-model.pt')
testfile = '../../data/interim/tags_few/sampled_20000_per_class/fasttext_distr/fasttext_test.txt'

with open(testfile, encoding='utf-8-sig') as f:
    lines = [line.rstrip('\n') for line in f]


labels = [int(''.join(filter(str.isdigit, text.split()[0]))) for text in lines]
trials = [text.split(' ', 1)[1] for text in lines]
print(type(labels[0]))
print(len(labels))
print(len(trials))

np.savetxt("transformers_labels.csv", labels, delimiter=",")

predictions = []
for trial in progress_bar(trials, "Getting predictions for : "):
    sentence = Sentence(trial)
    classifier.predict(sentence)
    # print(sentence.labels[0].value)
    # print(sentence.labels[0].score)
    predictions.append(int(sentence.labels[0].value))




np.savetxt("transformers_preds.csv", predictions, delimiter=",")

print(confusion_matrix(labels,predictions))        
print(classification_report(labels,predictions))

# sentence = Sentence(text)
# classifier.predict(sentence)
# print(sentence.labels)


#