# -*- coding: utf-8 -*-
import torch
from loss import FocalLoss

class Tuner(torch.nn.Module):
    
    def __init__(self):
        super(Tuner, self).__init__()
        pass
    
    def batch_config(self, trial):
        self.batch_size = trial.suggest_categorical('batch_size', [32])
        return self.batch_size
    
    def criterion_config(self, trial, device, sample_weight):
        # Cross Entropy, Weighted Cross Entropy, Focal Loss
        self.criterion_name = trial.suggest_categorical('criterion', ['CE','WCE','FL'])
        # Define criterion
        if self.criterion_name == 'CE':
            criterion = torch.nn.CrossEntropyLoss()
        if self.criterion_name == 'WCE':            
            criterion = torch.nn.CrossEntropyLoss(sample_weight.to(device))
        if self.criterion_name == 'FL':
            alpha = trial.suggest_discrete_uniform('alpha', 0.1, 1.0, 0.05)
            gamma = trial.suggest_discrete_uniform('gamma', 0.1, 2.0, 0.1)
            criterion = FocalLoss(alpha, gamma)
        return criterion
    
    def learning_rate_config(self, trial):
        self.learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)        
        return self.learning_rate
    
    def weight_decay_config(self, trial):         
        self.weight_decay = 0.0 if self.criterion_name != 'CE' else trial.suggest_categorical('weight_decay', [0.1,0.2,0.5,0.8,0.9])
        return self.weight_decay
    
    def optimizer_config(self, trial, model):
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam','SGD'])        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay_config(trial))
        if optimizer_name == 'SGD':
            momentum = trial.suggest_categorical('momentum', [0.1,0.2,0.8,0.9])
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay_config(trial), momentum=momentum)
        return optimizer
    
    def regularization_strength_config(self, trial):
        self.regularization_strength = trial.suggest_discrete_uniform('reg_strength', 0.5, 1.0, 0.05)
        return self.regularization_strength
    
    def layer_config(self, trial):
        self.n_layers = trial.suggest_int('n_layers', 1, 3)
        return self.n_layers
        
    def kernel_size_config(self, trial):
        self.kernel_size = trial.suggest_int('kernel_size', 3, 15, 3)
        return self.kernel_size

    def hidden_config(self, trial, i=0):
        # 4, 8, 16, 
        self.hidden = trial.suggest_categorical("hidden_{}".format(i), [64,128,256])
        return self.hidden
    
    def dropout_rate_config(self, trial, i=0):
        self.dropout = trial.suggest_discrete_uniform("dropout_{}".format(i), 0.0, 0.5, 0.05)
        return self.dropout
    
    def dense_config(self, trial):
        # 4096
        self.fc_1 = trial.suggest_categorical('fc_1', [1024, 2048])
        self.fc_2 = trial.suggest_categorical('fc_2', [512,1000])
        return self.fc_1, self.fc_2
    
    def tree_count(self, trial):
        self.num_tress = trial.suggest_categorical('num_trees', [64,128,256,1024,4096,16384])
        return self.num_tress
    
    def neighbours_count(self, trial):
        self.num_neighbors = trial.suggest_int('num_neighbours', 5, 55, 2)
        return self.num_neighbors