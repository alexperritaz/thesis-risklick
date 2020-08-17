# -*- coding: utf-8 -*-
import numpy as np
import lxml.etree as ET

from utils import walk_files, progress_bar, read_csv, write_csv

class RiskAssessment:
    
    def __init__(self, src_directory, src_pubmed, ct_status, study_type):   
        self.src_directory = src_directory
        self.medline_pubmed_results = read_csv(src_pubmed)
        self.ct_status = ct_status
        self.study_type = study_type
        self.filter_study = self.filter_type if self.study_type else self.filter_none
        self.data = []
        pass

    # Proceed to data extraction
    def process(self):
        filepaths = walk_files(self.src_directory)
        for filepath in progress_bar(filepaths, "Processing files : "):
            self.process_file(filepath)
        self.data = np.asarray(self.data)
        self.data = np.insert(self.data, 0, ['nct_id','classes'], 0)
        pass
    
    def process_file(self, filepath):
        tree = ET.parse(filepath)
        if self.filter_study(tree):
            name = self.get_name(tree)
            risk = self.get_risk(tree, name)
            self.data.append([name,risk])
        pass

    def get_name(self, tree):
        return tree.xpath('//id_info//nct_id/text()')[0].upper()
    
    # Get risk level of current CT
    def get_risk(self, tree, name):
        # ct_status[0] : CT On-going values
        # ct_status[1] : CT Terminated values
        overall_status = tree.xpath('//overall_status/text()')[0].lower()
        if overall_status == "completed" and self.results(tree, name) or overall_status == "approved for marketing": return 0 # return "insignificant"        
        if overall_status in self.ct_status[0]: return 1 # return "minor"
        if overall_status == "completed" and not self.results(tree, name): return 2 # return "moderate"
        if overall_status in self.ct_status[1] and self.results(tree, name): return 3 # return "major"
        if overall_status in self.ct_status[1] and not self.results(tree, name): return 4 # return "critical"        
        return -1 # return "unclassified"
    
    def results(self, tree, name):        
        if tree.xpath('//clinical_results') : return True
        if tree.xpath('//reference//PMID/text()') : return True
        if tree.xpath('//results_reference//PMID/text()') : return True
        if name in self.medline_pubmed_results : return True
        return False
    
    def to_onehot(self):
        self.data = np.delete(self.data, 0, 0)
        labels = np.array(self.data[:,-1], dtype=np.int)
        onehot_labels = np.zeros((labels.size, labels.max()+1))
        onehot_labels[np.arange(labels.size),labels] = 1
        self.data = np.delete(self.data, -1, axis=1)
        self.data = np.concatenate((self.data, onehot_labels), axis=1)
        self.data = np.insert(self.data, 0, ['nct_id','class 0','class 1','class 2','class 3','class 4'], 0)
        pass

    # Filter out non-interventionnal trials
    def filter_type(self, tree):
        return tree.xpath('//study_type/text()')[0].lower() in self.study_type
    
    # Filter - none
    def filter_none(self, tree):
        return True
    
    # Save files
    def save(self, dst_file):
        write_csv(dst_file, self.data, True)
        pass