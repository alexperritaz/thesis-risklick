# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import tqdm
tqdm.pandas()

DELIMITER = '|||'

# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))

class Encode:
    def __init__(self, src_file):
        self.src_file = src_file
        self.dataframe = pd.read_csv(self.src_file)
        self.n_columns = len(self.dataframe.columns)
        self.counter = 1
        self.encoded = 0
        self.info = {}
        self.updated = []
        pass

    def encode_features(self, threshold_label_encoding, threshold_multihot_encoding, ordinal_filter):
        self.ordinal_filter = ordinal_filter
        self.label_encoding(threshold_label_encoding)
        self.multihot_encoding(threshold_multihot_encoding)
        pass

    # Only encode ordinal values
    def label_encoding(self, threshold):
        self.dataframe.progress_apply(lambda col: self.label_encode(col, threshold), axis=0)
        pass

    #
    def label_encode(self, column, threshold):
        # Get label encoded representation
        column = column.astype('category')
        codes = column.cat.codes
        classes = column.cat.categories
        
        if column.name in self.ordinal_filter:    
            # Check if passes threshold            
            if len(classes) <= threshold:
                self.dataframe[column.name] = codes
                self.updated.append(column.name)
                # Update info for statistical analysis
        self.info.update({column.name : len(classes)})
        pass
    
    # Only encode non-ordinal values
    def multihot_encoding(self, threshold):
        self.dataframe.progress_apply(lambda col: self.multihot_encode(col, threshold), axis=0)        
        pass
    
    def multihot_encode(self, column, threshold):
        # If used with label encoding, check if not already encoded
        if column.name not in self.updated:            
            # Split column if cell contains multiple values
            column_split = column.astype(str).apply(lambda value: value.split(DELIMITER)) # list string containing each mesh term
            # Get multihot representation of the data
            mlb = MultiLabelBinarizer(sparse_output=True)
            multihot = mlb.fit_transform(column_split)
            classes = mlb.classes_
        
        
            # Check if passes threshold
            if len(classes) <= threshold:
                # Transform into datafram named accordingly to multihot representation
                df_multihot = self.transform(column.name, classes, multihot)
                # Add multihot encoding at end and drop column
                self.update_dataframe(df_multihot, column)
        
            # Update info for statistical analysis
            self.info.update({column.name : len(classes)})
        pass
    
    #
    def transform(self, featurename, classes, encoding):
        encoding_headers = [featurename + '[' + c + ']' for c in classes]
        encoded_df = pd.DataFrame(encoding.toarray(), columns=encoding_headers)
        return encoded_df
    
    #
    def update_dataframe(self, encoded_df, column):
        self.dataframe = pd.concat([self.dataframe,encoded_df], axis=1)
        self.dataframe.drop(column.name, axis=1, inplace=True)
        pass
    
    # Instead of fixing fixed value use percentage # NOT USED ATM #
    # def threshold(self, column):
    #     # Using ratio of unique values over total
    #     if column.dtypes == object :
    #         return 1. * column.nunique() / column.count() < self.threshold_cat_per        
    #     return False

    # Save to csv    
    def save(self, dst_file, dst_offcut, dst_analysis):
        print('Saving statistics')
        pd.DataFrame(self.info, index=['n_unique']).to_csv(dst_analysis, index=False)
        print('Saving encodings')
        self.dataframe.select_dtypes(include=np.number).astype(int).to_csv(dst_file,index=False)
        print('Saving remain')
        # Replace delimiter
        self.dataframe = self.dataframe.replace({'\|\|\|': '. '}, regex=True)
        self.dataframe.select_dtypes(exclude=np.number).astype(str).to_csv(dst_offcut, index=False)
        pass