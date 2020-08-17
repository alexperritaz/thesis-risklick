# -*- coding: utf-8 -*-
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer

import torch


class TransformerClassifier:

    def __init__(self):
        
        pass
    
    # def load_model(self):
    #     self.cl

    def load_corpus(self, src_directory):
        # Get the corpus
        self.corpus: Corpus = ClassificationCorpus(src_directory, test_file='fasttext_test.txt', dev_file='fasttext_dev.txt', train_file='fasttext_train.txt')
        # Create the label dictionary
        self.label_dict = self.corpus.make_label_dictionary()
    
    def load_settings(self):
        # Make a list of word embeddings        
        # word_embeddings = [TransformerWordEmbeddings('bert-base-uncased', layers='-1,-2,-3,-4', use_scalar_mix=True, pooling_operation="mean")]
        
        # self.document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased', layers='-1,-2,-3,-4', fine_tune=True)
        
        self.document_embeddings = DocumentPoolEmbeddings([WordEmbeddings('glove')], pooling='mean')
        
        # Initialize document embedding by passing list of word embeddings
        # self.document_embeddings = DocumentPoolEmbeddings(document_embeddings, pooling='mean') # hidden_size=256,
        
        self.optimizer = torch.optim.SGD
        
        self.params = {
            'learning_rate' : 3e-5, 
            # 'min_learning_rate' : 0.00001, 
            # 'mini_batch_size' : 256, # 8 
            # 'mini_batch_chunk_size' : 4,
            'anneal_factor' : 0.5, 
            'patience' : 5, 
            'max_epochs' : 50, 
            # 'anneal_with_restarts' : True
            }
        pass
    
    def train(self, dst_model):
        
        # Create the text classifier
        classifier = TextClassifier(self.document_embeddings, label_dictionary=self.label_dict)
        # Initialize the text classifier trainer
        trainer = ModelTrainer(classifier, self.corpus, optimizer=self.optimizer)
        # Start the training
        trainer.train(dst_model, **self.params)
        
        
if __name__ == '__main__':
    
    classifier = TransformerClassifier()
    classifier.load_corpus('../../data/interim/tags_few/sampled_20000_per_class/fasttext/')
    classifier.load_settings()
    # classifier.train('../models/tags_few_5000/bert_mean_encoded_2/Transformers/50_epoch_last_4/')
    classifier.train('../models/tags_few_20000/glove/Transformers/50_epoch_last_4/')
    pass