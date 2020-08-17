# -*- coding: utf-8 -*-
from tuning import Tuner

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from model_deep import CNNDynamic, LSTM, RNN, GRU



class Switcher(object):
    
    def __init__(self, trial, device, num_features, num_classes, direction, training_score_path, validation_score_path, training_loss_path, validation_loss_path):
        self.trial = trial
        self.device = device
        self.num_features = num_features
        self.num_classes = num_classes
        self.direction = direction
        self.training_score_path = training_score_path
        self.validation_score_path = validation_score_path
        self.training_loss_path = training_loss_path
        self.validation_loss_path = validation_loss_path 
        pass
    
    def CNN(self, max_iter=200):
        tuning = Tuner()
        layer_sizes = tuple()
        dropout_rates = tuple()
        num_layers = tuning.layer_config(self.trial)
        for i in range(num_layers):
            layer_sizes += (tuning.hidden_config(self.trial, i),) 
            dropout_rates += (tuning.dropout_rate_config(self.trial, i),)
        
        params = {
            # Instances
            'tuner' : tuning,
            'trial' : self.trial,
            'device': self.device,
            # Model settings
            'num_features'          : self.num_features,
            'num_classes'           : self.num_classes,            
            'dropout_rates'         : dropout_rates,            
            'hidden_layer_sizes'    : layer_sizes,
            'dense_layer_sizes'     : tuning.dense_config(self.trial),
            'kernel_size'           : tuning.kernel_size_config(self.trial),
            'stride' : 1,
            # Learning settings
            'direction'             : self.direction,
            'early_stopping'        : True, 
            'max_iter'              : max_iter,
            'n_iter_no_change'      : 10, 
            'validation_fraction'   : 0.25,
            'verbose'               : True,
            # Learning parameters
            'batch_size'            : tuning.batch_config(self.trial),
            'learning_rate_init'    : tuning.learning_rate_config(self.trial),
            # 'weight_decay'          : tuning.weight_decay_config(self.trial)
            
            'training_score_path'   : self.training_score_path,
            'validation_score_path' : self.validation_score_path,
            'training_loss_path'    : self.training_loss_path,
            'validation_loss_path'  : self.validation_loss_path,
            
            }
        
        model = CNNDynamic(params)
        model.to(self.device)
        return model
        pass
    
    def RNN(self, max_iter=200):
        tuning = Tuner()
        params = {
            # Instances
            'tuner'                 : tuning,
            'trial'                 : self.trial,
            'device'                : self.device,
            # Model settings
            'hidden_size'           : tuning.hidden_config(self.trial, 0),
            'num_features'          : self.num_features,
            'num_classes'           : self.num_classes,
            'num_layers'            : tuning.layer_config(self.trial),
            'dropout_rate'          : tuning.dropout_rate_config(self.trial, 0),
            # Learning settings
            'direction'             : self.direction,
            'early_stopping'        : True, 
            'max_iter'              : max_iter,
            'n_iter_no_change'      : 10, 
            'validation_fraction'   : 0.25,
            'verbose'               : True,
            # Learning parameters
            'batch_size'            : 256, # tuning.batch_config(self.trial),          
            'learning_rate_init'    : tuning.learning_rate_config(self.trial),
            # 'weight_decay'          : tuning.weight_decay_config(self.trial),
            
            'training_score_path'   : self.training_score_path,
            'validation_score_path' : self.validation_score_path,
            'training_loss_path'    : self.training_loss_path,
            'validation_loss_path'  : self.validation_loss_path,
            }        
        model = RNN(params)
        model.to(self.device)
        return model
    
    def LSTM(self, max_iter=200):
        tuning = Tuner()
        params = {
            # Instances
            'tuner'                 : tuning,
            'trial'                 : self.trial,
            'device'                : self.device,
            # Model settings
            'hidden_size'           : tuning.hidden_config(self.trial, 0),
            'num_features'          : self.num_features,
            'num_classes'           : self.num_classes,
            'num_layers'            : tuning.layer_config(self.trial),
            'dropout_rate'          : tuning.dropout_rate_config(self.trial, 0),
            # Learning settings
            'direction'             : self.direction,
            'early_stopping'        : True, 
            'max_iter'              : max_iter,
            'n_iter_no_change'      : 10, 
            'validation_fraction'   : 0.25,
            'verbose'               : True,
            # Learning parameters
            'batch_size'            : 256, # tuning.batch_config(self.trial),       
            'learning_rate_init'    : tuning.learning_rate_config(self.trial),
            # 'weight_decay'          : tuning.weight_decay_config(self.trial),
            
            'training_score_path'   : self.training_score_path,
            'validation_score_path' : self.validation_score_path,
            'training_loss_path'    : self.training_loss_path,
            'validation_loss_path'  : self.validation_loss_path,
            }        
        model = LSTM(params)
        model.to(self.device)
        return model
    
    def GRU(self, max_iter=200):
        tuning = Tuner()
        params = {
            # Instances
            'tuner'                 : tuning,
            'trial'                 : self.trial,
            'device'                : self.device,
            # Model settings
            'hidden_size'           : tuning.hidden_config(self.trial, 0),
            'num_features'          : self.num_features,
            'num_classes'           : self.num_classes,
            'num_layers'            : tuning.layer_config(self.trial),
            'dropout_rate'          : tuning.dropout_rate_config(self.trial, 0),
            # Learning settings
            'direction'             : self.direction,
            'early_stopping'        : True, 
            'max_iter'              : max_iter,
            'n_iter_no_change'      : 10, 
            'validation_fraction'   : 0.25,
            'verbose'               : True,
            # Learning parameters
            'batch_size'            : 256, # tuning.batch_config(self.trial),      
            'learning_rate_init'    : tuning.learning_rate_config(self.trial),
            # 'weight_decay'          : tuning.weight_decay_config(self.trial),
            
            'training_score_path'   : self.training_score_path,
            'validation_score_path' : self.validation_score_path,
            'training_loss_path'    : self.training_loss_path,
            'validation_loss_path'  : self.validation_loss_path,
            }        
        model = GRU(params)
        model.to(self.device)
        return model
    
    def OneVsLogReg(self, max_iter=7600):
        tuning = Tuner()
        params = {
            'C'             : tuning.regularization_strength_config(self.trial),
            'max_iter'      : 7600,
            'multi_class'   : 'ovr',
            'verbose'       : 1,
            'n_jobs' : 8
            }
        # multi_class{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
        # C : Inverse of regularization strength - Lower means stronger
        # default solver='lbfgs' 
        # model = OneVsRestClassifier(LogisticRegression(C=C, max_iter=max_iter, verbose=1))
        model = LogisticRegression(**params)
        return model
    
    def OneVsSVM(self, max_iter=7600):
        tuning = Tuner()
        params = {
            'C'         : tuning.regularization_strength_config(self.trial),
            'max_iter'  : 7600,
            'verbose'   : True,
            }
        # C : Inverse of regularization strength - Lower means stronger
        model = OneVsRestClassifier(SVC(**params))
        return model
    
    def LogReg(self, max_iter=200):
        tuning = Tuner()
        params = {
            'C'         : tuning.regularization_strength_config(self.trial),
            'max_iter'  : max_iter,
            'verbose'   : 1,
            'n_jobs' : 8
            }
        # C : Inverse of regularization strength - Lower means stronger
        model = LogisticRegression(**params)
        return model
    
    def MLP(self, max_iter=200):
        tuning = Tuner()
        layers = tuple()
        num_layers = tuning.layer_config(self.trial)  
        for i in range(num_layers):
            layers += (tuning.hidden_config(self.trial, i),) 

        params = {
            'activation'            : 'relu',
            'early_stopping'        : True, 
            'hidden_layer_sizes'    : layers,
            'learning_rate_init'    : tuning.learning_rate_config(self.trial),
            'max_iter'              : max_iter,
            'n_iter_no_change'      : 10, 
            'random_state'          : 21,
            'solver'                : 'adam',
            'validation_fraction'   : 0.25, 
            'verbose'               : True,
            }
        
        model = MLPClassifier(**params)
        return model
    
    def KNN(self, max_iter=200):
        tuning = Tuner()
        params = {
            'n_neighbors' : tuning.neighbours_count(self.trial),
            'n_jobs' : 8
            }
        print(params)
        model = KNeighborsClassifier(**params)
        return model
    
    def RF(self, max_iter=200):
        tuning = Tuner()
        params = {
            'n_estimators'    : 16384,#tuning.tree_count(self.trial),
            'class_weight'  : 'balanced',
            'verbose'       : 1,
            'criterion'     : 'entropy',
            'n_jobs' : 8
            }
        model = RandomForestClassifier(**params)
        return model
    
    def GDB(self, max_iter=200):
        tuning = Tuner()
        params = {
            'n_estimators'          : 200,
            'n_iter_no_change'      : 10, 
            'learning_rate'         : tuning.learning_rate_config(self.trial),
            'validation_fraction'   : 0.25, 
            'verbose'               : 1,
            }
        model = GradientBoostingClassifier(**params)
        return model