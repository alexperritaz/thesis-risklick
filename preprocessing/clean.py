# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd

from fastnumbers import fast_int

class Clean:

    def __init__(self, src_file):   
        self.src_file = src_file
        self.dataframe = pd.read_csv(self.src_file)
        pass

    # Remove handpicked columns    
    def filter_out(self, filter_list):
        for entry in filter_list :
            cols = [col for col in self.dataframe.columns if entry in col]
            discard = [col for col in cols if col not in filter_list [entry]]
            self.dataframe.drop(discard, axis=1, inplace=True)
        pass

    # Try to convert all to integers
    def str_to_int(self):
        # Transform str(int) into int
        self.dataframe = self.dataframe.applymap(fast_int)
        pass

    # Replace all null values
    def replace_nan(self):
        # Fill empty cells by -1
        self.dataframe = self.dataframe.fillna(-1)
        # Replace N/A by -1
        self.dataframe = self.dataframe.replace('N/A', -1)
        self.dataframe = self.dataframe.replace('N / A',-1)
        pass

    # Remove irrelevant columns
    def remove_default(self):
        # Remove column containing only default values
        [self.dataframe.drop(column, axis=1, inplace=True) for column in self.dataframe if len(self.dataframe[column].value_counts()) <= 1]
        # Remove column containing only different values - Here only applies to Trial ID ! Do not consider to apply in everycase
        [self.dataframe.drop(column, axis=1, inplace=True) for column in self.dataframe if len(self.dataframe[column].value_counts()) == len(self.dataframe.index)]
        pass
    
    def data_imputation(self, column_threshold=0.5, row_threshold=0.3):
        self.discard_columns = []
        self.discard_rows = []
        # Create list of indexes to imputate
        for i, column in enumerate(self.dataframe.columns):
            column_nan = self.dataframe[column].value_counts()[-1]
            if column_nan > len(self.dataframe) * column_threshold:
                self.discard_columns.append(i)
        # Create list of indexes to imputate
        for i, row in enumerate(self.dataframe.itertuples()):
            row_nan = row.count(-1)
            if row_nan > len(self.dataframe.columns) * row_threshold:
                self.discard_rows.append(i)
        # Impute date from in lists
        self.dataframe.drop(self.dataframe.columns[self.discard_columns], axis=1, inplace=True)
        self.dataframe.drop(self.dataframe.index[self.discard_rows], inplace=True)
        pass
        
    # Format all ages format into days
    def age_to_days(self):
        # Select age related columns        
        maximum_age = '/clinical_study/eligibility/maximum_age'
        minimum_age = '/clinical_study/eligibility/minimum_age'
        # Format age to number of days 
        # Days being the highest meaningful granularity
        # Removed hours / minutes
        self.dataframe[maximum_age] = self.dataframe[maximum_age].apply(self.format_age)
        self.dataframe[minimum_age] = self.dataframe[minimum_age].apply(self.format_age)    
        pass

    
    # Format age to number of days
    def format_age(self, value):    
        str_age = str(value).lower()
        days = self.remove_string(str_age)
        if any(word in str_age for word in ['years','year']): return days * 360
        elif any(word in str_age for word in ['months','month']): return days * 30
        elif any(word in str_age for word in ['weeks','week']): return days * 7
        elif any(word in str_age for word in ['days','day']): return days
        else: return -1

        
    # Remove string from age variable
    def remove_string(self, str_age):
        return int(re.sub("[^0-9]", "", str_age))
        
    # Format date to timestamp    
    def date_to_timestamp(self):
        # Identify datetime column names
        tmp_df = self.dataframe.apply(lambda col: pd.to_datetime(col, errors='ignore') if self.dataframe[col.name].dtypes == object else col, axis=0)
        col_df = tmp_df.select_dtypes(include=['datetime64[ns]'])
        # Apply transformation to timestamp from date columns                
        self.dataframe[col_df.columns] = self.dataframe[col_df.columns].applymap(lambda x: self.timestamp_conversion(x) if x != '-1' else x) 
        pass

    # Apply timestamp conversion
    def timestamp_conversion(self, str_date):
        return pd.to_datetime(str_date).value // 10 ** 9
        pass

    # Save file
    def save(self, dst_file):
        # Save Dataframe to csv
        self.dataframe.to_csv(dst_file,index=False)
        pass