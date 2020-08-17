# -*- coding: utf-8 -*-

import nltk
import numpy as np
import pandas as pd
from flair.data import Sentence
from deepsegment import DeepSegment
from flair.embeddings import DocumentPoolEmbeddings
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktLanguageVars

class BulletPointLangVars(PunktLanguageVars):
    sent_end_chars = ('.', '-', '•', '*', 'o', '∙', '‣', '◦')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EmbedBERT:
    
    def __init__(self, src_file, dst_file, dst_log):
        self.src_file = src_file
        self.dst_file = dst_file
        self.dst_log = dst_log
        self.dataframe = pd.read_csv(self.src_file)
        self.counter = 0
        pass
    
    def define(self, embedding):
        self.document_embeddings = embedding
        self.tokenizer = PunktSentenceTokenizer(lang_vars = BulletPointLangVars())
        self.tokenizer_punktless = DeepSegment('en')
        pass
    
    def process(self):
        print(self.dataframe.dtypes)        
        self.dataframe.apply(lambda col: self.embed_column(col) if self.dataframe[col.name].dtypes == object else col, axis=0)
        pass
    
    def embed_column(self, column):
        print(column.name)
        self.counter = self.counter + 1
        
        embeddings = column.astype(str).progress_apply(self.average_text_sentences)
        embedded_df = self.transform(embeddings, column.name)
        self.update_dataframe(embedded_df, column)
        pass
    
    def average_text_sentences(self, text):
        current_embeddings = []
        sentences = self.tokenizer.tokenize(text)
        if len(sentences) <= 1:
            sentences = self.tokenizer_punktless.segment_long(text)
        for sentence in sentences:
            sentence = sentence.rstrip('.-•*o∙‣◦')
            if len(sentence) == 0: continue
            sentence = Sentence(sentence)
            self.document_embeddings.embed([sentence])
            current_embeddings.append(sentence.embedding.data.detach().cpu().numpy())
        return np.average(np.asarray(current_embeddings), axis=0)
    
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