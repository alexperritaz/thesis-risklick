# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 19:55:27 2020

@author: Alex
"""

import pandas as pd

from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings

class Embed:
    
    def __init__(self, src_file, dst_file, dst_log):
        self.src_file = src_file
        self.dst_file = dst_file
        self.dst_log = dst_log
        self.dataframe = pd.read_csv(self.src_file)
        self.counter = 0
        pass
    
    def define(self, embedding_list):
        self.document_embeddings = DocumentPoolEmbeddings(embedding_list)
        pass
    
    def process(self):
        print(self.dataframe.dtypes)        
        self.dataframe.apply(lambda col: self.embed_column(col) if self.dataframe[col.name].dtypes == object else col, axis=0)
        pass
    
    def embed_column(self, column):
        print(column.name)
        self.counter = self.counter + 1
        
        embeddings = column.astype(str).progress_apply(self.embed_sentence)
        embedded_df = self.transform(embeddings, column.name)
        self.update_dataframe(embedded_df, column)
        pass
    
    def embed_sentence(self, text):
        sentence = Sentence(text)          
        self.document_embeddings.embed(sentence)
        return sentence.get_embedding().data.detach().cpu().numpy()
    
    def transform(self, embeddings, featurename):
        embedding_size = len(embeddings[0])
        embedding_header = [featurename + '['+ str(dim) +']' for dim in range(embedding_size)]
    
        embedded_df = pd.DataFrame.from_records(embeddings)
        embedded_df.columns = embedding_header
        return embedded_df
    
    def update_dataframe(self, embedded_df, column):
        self.dataframe = pd.concat([self.dataframe,embedded_df], axis=1)
        self.dataframe.drop(column.name, axis=1, inplace=True)
        pass
    
    def save(self):
        self.dataframe.to_csv(self.dst_file, index=False)
        pass