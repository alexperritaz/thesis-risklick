# -*- coding: utf-8 -*-
import os

import torch
import numpy as np
import pandas as pd

from pathlib import Path

from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader

class Dataset(object):
    def __init__(self):
        working_directory = str(Path(*Path(os.getcwd()).parts[:-1]))
        self.project_directory = os.path.expanduser(os.path.dirname(working_directory))
        pass      
    
    def file_path(self, configuration, embedding_name, embedding_pooling, encoding, tree_depth, sample_size):
        
        self.files = {
            'encodings'     : self.project_directory + './data/interim/%s/sampled_%s_per_class/encoded.csv' % (configuration, sample_size),
            'embeddings'    : self.project_directory + './data/interim/%s/sampled_%s_per_class/embedded_%s_%s.csv' % (configuration, sample_size, embedding_name, embedding_pooling),
            'labels'        : self.project_directory + './data/interim/%s/sampled_%s_per_class/labels_encoded.csv' % (configuration, sample_size)
            }
        
        return self.files
    
    
    def model_path(self, configuration, sample_size, embedding_name, embedding_pooling, encoding, tree_depth, model):        
        self.config_directory = self.project_directory + './scripts/models/%s_%s/%s_%s_%s_%s/' % (configuration, sample_size, embedding_name, embedding_pooling, encoding, tree_depth)
        self.model_directory = self.config_directory + '%s/' % (model)
        Path(self.model_directory).mkdir(parents=True, exist_ok=True) 
        
        self.model_files = {
            'train_split'           : self.config_directory  + 'train_split.csv',
            'validation_split'      : self.config_directory  + 'valid_split.csv',
            'test_split'            : self.config_directory  + 'test_split.csv',
            #
            'study_params'          : self.model_directory + 'study_params.csv',
            'study_optuna'          : self.model_directory + 'study_optuna.db',
            'best_model'            : self.model_directory + 'best_model.pickle'
            }
        return self.model_files
    
    def trial_path(self, trial_number):
        self.trial_directory = self.model_directory + 'trial_%d/' % (trial_number)
        Path(self.trial_directory).mkdir(parents=True, exist_ok=True)         
        self.trial_files = {
            'training_loss'         : self.trial_directory + 'training_loss.csv',
            'validation_loss'       : self.trial_directory + 'validation_loss.csv',
            'training_score'        : self.trial_directory + 'training_accuracy.csv',
            'validation_score'      : self.trial_directory + 'validation_accuracy.csv',
            'y_pred'                : self.trial_directory + 'y_pred.csv',
            'y_true'                : self.trial_directory + 'y_true.csv',
            }
        return self.trial_files
            
    def save_model(self, model_name, model):
        # path = self.project_directory + './trainings/%s/%s/%s/%s' % (self.configuration, self.embedding_config, self.pooling_type, model_name, model)
        # torch.save(model, path)
        pass
    
    def load_model(self):
        # model = TheModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(PATH))
        # model.eval()
        pass
    
    def class_distribution(self, numpy_array):
        unique, counts = np.unique(numpy_array, return_counts=True)
        count_dict = dict(zip(unique, counts))
        return count_dict
    
    def class_weights(self, y):        
        class_count = [i for i in self.class_distribution(y).values()]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
        return class_weights
    
    def class_weight_sampler(self, dataset, class_weights):
        class_weights_all = class_weights[self.class_list(dataset)]
        weighted_sampler = WeightedRandomSampler(weights=class_weights_all, num_samples=len(class_weights_all), replacement=True)
        return weighted_sampler

    def class_list(self, dataset):
        labels = []
        for _, label in dataset:
            labels.append(label) 
        labels = torch.tensor(labels)
        labels = labels[torch.randperm(len(labels))]
        return labels
    
    pass
    
    
class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data.flatten()
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)