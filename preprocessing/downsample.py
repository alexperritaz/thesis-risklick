# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 18:15:54 2020

@author: Alex
"""


import pandas as pd
import random
import shutil
from utils import walk_files, progress_bar

class DownSample:
    
    def __init__(self, src_directory, dst_directory, src_labels, sample_size):
        self.src_directory = src_directory
        self.src_labels = src_labels
        self.dst_directory = dst_directory
        self.sample_size = sample_size
        pass

    def downsize(self, dst_file):
        df = pd.read_csv(self.src_labels)
        n_classes = df['classes'].nunique()
        filepaths = walk_files(self.src_directory)        
        seed = 42
        nct_ids = []
        
        
        if self.sample_size != 'all':
            for i in range(n_classes):
                classes_i = df.loc[df['classes'] == i]
                sample_i = classes_i.sample(self.sample_size, random_state=seed)
                sorted_i = sorted(sample_i['nct_id'])
                for path in progress_bar(filepaths, "Extracting class %s : " % (i)):
                    if any(nct_id in path for nct_id in sorted_i):
                        shutil.copy2(path, self.dst_directory)
                nct_ids.extend(sorted_i)
            df[df['nct_id'].isin(nct_ids)].to_csv(dst_file, index=False)
        else :
            for path in progress_bar(filepaths, "Processing files : "):
                shutil.copy2(path, self.dst_directory)      
            df[df['nct_id'].isin(nct_ids)].to_csv(dst_file, index=False)
        pass
    
    