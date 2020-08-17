# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from editdistance import eval as levenshtein

from sklearn.preprocessing import KBinsDiscretizer


tqdm.pandas()

DELIMITER = '|||'
MESH_DELIMITER = '|'

class Transform:
    
    def __init__(self, src_file):
        self.src_file = src_file
        self.dataframe = pd.read_csv(self.src_file)
        pass

    
    def define_columns(self): 
        # Age related columns        
        self.maximum_age = '/clinical_study/eligibility/maximum_age'
        self.minimum_age = '/clinical_study/eligibility/minimum_age'
        # Start and end of trial related columns
        self.start_date = '/clinical_study/start_date'
        self.end_date = '/clinical_study/completion_date'
        # Duration related column
        self.duration = 'duration'        
        # Enrollment related column
        self.enrollment = '/clinical_study/enrollment'        
        # Mesh related column
        self.mesh_terms = '/clinical_study/condition_browse/mesh_term'
        pass
    
    # Convert age from days to years
    def convert_age(self):   
       self.dataframe[self.maximum_age] = self.dataframe[self.maximum_age].apply(lambda x: math.ceil(x / 360) if x != -1 else x)
       self.dataframe[self.minimum_age] = self.dataframe[self.minimum_age].apply(lambda x: math.ceil(x / 360) if x != -1 else x)        
       pass
        
    def convert_date(self):        
        # Mask containing rows that are valid (both valid timestamps)
        tmp_df = self.dataframe[[self.start_date, self.end_date]]
        date_mask = (tmp_df != -1).all(axis=1)
        # Create duration column with default value to -1
        self.dataframe[self.duration] = -1
        # Update values of duration (in days)
        self.dataframe.loc[date_mask,self.duration] = ((self.dataframe[self.end_date] - self.dataframe[self.start_date]) / 86400).astype(int)
        # Drop the columns used to calculate duration
        self.dataframe.drop([self.start_date, self.end_date], axis=1, inplace=True)
        pass
    
    # Bin data by group
    def bin_data(self):
        duration_range = [0,30,90,180,360,720,1800,3600,10800,36000]
        duration_label = ['0-1 month','1-3 months','3-6 months','6-12 months','1-2 years','2-5 years','5-10 years','10-30 years','30-100 years']
        self.bin_column(self.duration, duration_range, duration_label)
        
        enrollment_range = [0,25,50,100,200,500,1000,5000,99999999]
        enrollment_label = ['0-25','25-50','50-100','100-200','200-500','500-1000','1000-5000','5000-99999999']
        self.bin_column(self.enrollment, enrollment_range, enrollment_label)
        
        age_range = [0,18,65,200]
        maximum_label = [18,65,200]
        minimum_label = [0,18,65]
        self.bin_column(self.maximum_age, age_range, maximum_label)
        self.bin_column(self.minimum_age, age_range, minimum_label)
    
    # Bin each column accordingly to group
    def bin_column(self, col, custom_range, custom_label):
        self.dataframe[col] = self.dataframe[col].replace(-1,np.nan)
        self.dataframe[col] = pd.cut(self.dataframe[col], custom_range, labels=custom_label)
        self.dataframe[col] = self.dataframe[col].cat.add_categories(-1)
        self.dataframe[col] = self.dataframe[col].fillna(-1)
        pass
        
    # Elevate mesh terms in tree accordingly
    def meshify_data(self, base_file, ref_file, segment_len, tree_depth):                       
        base_df = pd.read_csv(base_file)
        ref_df = pd.read_csv(ref_file)
        self.dataframe[self.mesh_terms] = self.dataframe[self.mesh_terms].progress_apply(lambda x: self.meshify(x, base_df, ref_df, tree_depth, segment_len) if x != '-1' else x)
        pass
       
    # Elevate mesh terms in tree accordingly to specified depth
    def meshify(self, str_terms, base_df, ref_df, tree_depth, segment_len):
        # Trial mesh terms
        mesh_terms = []        
        list_mesh_reference = []        
        # All of the mesh terms
        all_mesh = base_df.columns        
        # Split trial mesh terms
        terms = str_terms.split(DELIMITER)
        # Define char length to keep based off the tree level 
        len_reference = tree_depth * segment_len - 1
        # For each mesh term in this cell
        for mesh in terms:
            try:
                # Get list of mesh numbers
                list_mesh_reference = base_df[mesh][0].split(MESH_DELIMITER)
            except:
                # Levenshtein distance - get most similar word in case of mispelling
                lev_mesh = min(all_mesh, key=lambda target: levenshtein(target, mesh))
                # Get list of mesh numbers based off the corrected mesh
                list_mesh_reference = base_df[lev_mesh][0].split(MESH_DELIMITER)
                pass            
            # For each element of sub mesh list
            for reference in list_mesh_reference:
                # Find match in ref_df
                generalized_mesh = ref_df[reference[0:len_reference]][0]
                # Add higher level mesh to list
                mesh_terms.append(generalized_mesh) if generalized_mesh not in mesh_terms else ''
            pass
        # Return string formatted using delimiter
        return DELIMITER.join(mesh_terms)
    
    # Save file
    def save(self, dst_file):
        # Save Dataframe to csv
        self.dataframe.to_csv(dst_file, index=False)
        pass