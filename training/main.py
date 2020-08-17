# -*- coding: utf-8 -*-
from tuning import Tuner

import os
import optuna
import pickle
import torch
import numpy as np
import pandas as pd

from os import path
from model_switcher import Switcher
from model import Model

import random

MAX_EPOCHS = 200

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve

from dataset import Dataset

class Main(object):
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'   
        pass
    
    def load_data(self, configuration, embedding_name, embedding_pooling, encoding, tree_depth, sample_size, n_trials):
        
        # Load data
        print('Loading embeddings data ... ')
        self.embeddings = pd.read_csv(self.file_paths['embeddings'])
        print('Loading encodings data ... ')
        self.encodings = pd.read_csv(self.file_paths['encodings'])
        print('Loading labels ... ')
        self.labels = pd.read_csv(self.file_paths['labels'])

        # Define data
        self.data_x = np.concatenate((self.encodings,self.embeddings),axis=1)
        self.data_y = np.array(self.labels.to_numpy()[:,1:], dtype=np.float)

        # Print class distribution
        print('Class distribution : ' + str(self.dataset.class_distribution(self.data_y)))

        # Split into train-val and test
        # Stratify to keep same distribution in train and test
        self.X_trainval, self.X_test, self.y_trainval, self.y_test = train_test_split(self.data_x, self.data_y, test_size=0.2, stratify=self.data_y, random_state=69)
        # Only for saving splits, same splits used in model.py
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_trainval, self.y_trainval, test_size=0.25, stratify=self.y_trainval, random_state=21)
        
        # Save splits
        if not path.exists(self.model_paths['test_split']):
            pd.DataFrame(np.concatenate((self.X_train, self.y_train), axis=1)).to_csv(self.model_paths['train_split'], index=False, header=False)
            pd.DataFrame(np.concatenate((self.X_val, self.y_val), axis=1)).to_csv(self.model_paths['validation_split'], index=False, header=False)
            pd.DataFrame(np.concatenate((self.X_test, self.y_test), axis=1)).to_csv(self.model_paths['test_split'], index=False, header=False)
        
        # Data properties        
        self.num_features = self.data_x.shape[1]   
        self.num_classes = 5 # self.data_y.shape[1]

        pass
    
    #
    def select_model(self, trial, model_name):
        switch = Switcher(trial, self.device, self.num_features, self.num_classes, self.direction, self.trial_paths['training_score'], self.trial_paths['validation_score'], self.trial_paths['training_loss'], self.trial_paths['validation_loss'])
        models = {
            'LogReg' : switch.LogReg,
            'MLP' : switch.MLP,
            'KNN' : switch.KNN,
            'RF'  : switch.RF,
            'GDB' : switch.GDB,
            'OneVsLogReg' : switch.OneVsLogReg,
            'OneVsSVM'    : switch.OneVsSVM,
            'CNN' : switch.CNN,
            'RNN' : switch.RNN,
            'LSTM': switch.LSTM,
            'GRU' : switch.GRU,
            }
        return models.get(model_name, lambda: 'Invalid model')(max_iter = 300)
    
    #
    def objective(self, trial):
        # Get all filepaths for current trial
        self.trial_paths = self.dataset.trial_path(trial.number)
        
        # try:
        # Get the model and fit the data
        self.model = self.select_model(trial, self.model_name)        
        self.model.fit(self.X_trainval, self.y_trainval.reshape(-1))
        # except:
        #     print('CUDA out of memory : Trial Prunned')
        #     raise optuna.TrialPruned()
        # Get predictions
        y_pred = self.model.predict(self.X_test)        
        # Save predictions
        pd.DataFrame(y_pred).to_csv(self.trial_paths['y_pred'], index=False, header=False)
        pd.DataFrame(self.y_test).to_csv(self.trial_paths['y_true'], index=False, header=False)
        
        # Print overall stats at end of trial
        print(confusion_matrix(self.y_test,y_pred))        
        print(classification_report(self.y_test,y_pred))
        # Get accuracy
        report_dict = classification_report(self.y_test,y_pred, output_dict=True)        
        accuracy = report_dict.get('accuracy')

        # Only for deep models
        if type(self.model).__module__ == 'model_deep':
            # Save learning curves
            self.model.save_stats()
            pass
        
        # Save model
        if accuracy > self.best_value:
            self.best_model = self.model
            self.best_value = accuracy
        
        return accuracy
    
    #
    def tune_model(self, configuration, sample_size, embedding_name, embedding_pooling, encoding, tree_depth, model, n_trials):
        # Classifier used within this study
        self.dataset = Dataset()
        self.file_paths = self.dataset.file_path(configuration, embedding_name, embedding_pooling, encoding, tree_depth, sample_size)
        self.model_paths = self.dataset.model_path(configuration, sample_size, embedding_name, embedding_pooling, encoding, tree_depth, model)
        self.load_data(configuration, embedding_name, embedding_pooling, encoding, tree_depth, sample_size, n_trials)
        
        # Study properties
        self.model_name = model
        self.direction = 'maximize'
        self.best_value = 0
        
        # Optuna sqlite storage path        
        storage_path = 'sqlite:///' + self.model_paths['study_optuna']        
        
        # Create new study for each model
        self.study = optuna.create_study(direction=self.direction, study_name=model, storage=storage_path, load_if_exists=True)
        self.study.optimize(self.objective, n_trials=n_trials)
        
        with open(self.model_paths['best_model'], "wb") as file:
            pickle.dump(self.best_model, file, protocol=4)
        
        self.study.trials_dataframe().to_csv(self.model_paths['study_params'], index=False, header=False)         
        
        # Recap after trials
        print("Best params", self.study.best_params)
        print("Best trial", self.study.best_trial)
        print("Best value", self.study.best_value)  
        pass

#
if __name__ == "__main__":
    main = Main()    
    
    settings = ('tags_few','20000','glove','avg','encoded','3')
    
    # Base models
    main.tune_model(*settings, 'RF', 1)
    # main.tune_model(*settings, 'KNN', 10)
    # main.tune_model(*settings, 'GDB', 5)
    # main.tune_model(*settings, 'LogReg', 10)
    # main.tune_model(*settings, 'OneVsLogReg', 10)
    # main.tune_model(*settings, 'OneVsSVM', 10)
    # # MLP
    # main.tune_model(*settings, 'MLP', 20)
    # Deep learning
    # main.tune_model(*settings, 'CNN', 50)
    # main.tune_model(*settings, 'RNN', 25)
    # main.tune_model(*settings, 'LSTM', 25)
    # main.tune_model(*settings, 'GRU', 25)
    
    # Other Data
    # settings = ('tags_few','20000','bert','avg','encoded','2', 100)
    # main.load_data(*settings)
    # main.tune_model(*settings, 'LogReg')
    # main.tune_model(*settings, 'MLP')
    # main.tune_model(*settings, 'CNN')
    # main.tune_model(*settings, 'LSTM')
    pass