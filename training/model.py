# -*- coding: utf-8 -*-
import torch
import optuna
import numpy as np
import pandas as pd
from utils import progress_bar
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from early_stopping import EarlyStopping

from sklearn.model_selection import train_test_split

from dataset import ClassifierDataset, Dataset

class Model(torch.nn.Module):
    
    def __init__(self, kwargs):
        super(Model, self).__init__()
                
        if 'batch_size' in kwargs: self.batch_size = kwargs['batch_size']
        if 'device' in kwargs: self.device = kwargs['device']
        if 'direction' in kwargs: self.direction = kwargs['direction']
        if 'max_iter' in kwargs: self.max_iter = kwargs['max_iter']
        if 'n_iter_no_change' in kwargs: self.n_iter_no_change = kwargs['n_iter_no_change']
        if 'learning_rate_init' in kwargs: self.learning_rate = kwargs['learning_rate_init']
        if 'trial' in kwargs: self.trial = kwargs['trial']
        if 'tuner' in kwargs: self.tuning = kwargs['tuner']
        if 'validation_fraction' in kwargs: self.validation_fraction = kwargs['validation_fraction']
        if 'weight_decay' in kwargs: self.weight_decay = kwargs['weight_decay']
        
        if 'training_score_path' in kwargs: self.training_score_path = kwargs['training_score_path']
        if 'validation_score_path' in kwargs: self.validation_score_path = kwargs['validation_score_path']
        if 'training_loss_path' in kwargs: self.training_loss_path = kwargs['training_loss_path']
        if 'validation_loss_path' in kwargs: self.validation_loss_path = kwargs['validation_loss_path']
        
        self.early_stopping = EarlyStopping(self.direction, self.n_iter_no_change)
        
        pass
    
    def ingest_data(self, X_trainval, y_trainval):
        
        dataset = Dataset()
        
        # Split train into train-val
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=21)
        
        train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())        
        
        self.class_weights = dataset.class_weights(y_train)
        
        self.weighted_sampler = dataset.class_weight_sampler(train_dataset, self.class_weights)
        
        self.train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=self.batch_size, sampler=self.weighted_sampler)
        self.validation_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size)
        pass
    
    def fit(self, data_x, data_y):      
        
        self.accuracy_stats = {
            'train'         : [],
            'validation'    : []
        }

        self.loss_stats = {
            'train'         : [],
            'validation'    : []
        }

        self.ingest_data(data_x, data_y.reshape(-1,1))
        
        self.criterion = self.tuning.criterion_config(self.trial, self.device, self.class_weights)
        self.optimizer = self.tuning.optimizer_config(self.trial, self.model)
        
        print(self.trial.params)
        
        # for self.epoch in progress_bar(range(self.max_iter), "\nEpoch progression : "):
        for self.epoch in range(self.max_iter):
            train_epoch_loss = 0
            train_epoch_acc = 0
            self.model.train()
            for inputs, targets in self.train_loader:
                # Reset gradients
                self.optimizer.zero_grad()
                # Pass inputs and targets to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                # Get model output
                outputs = self.forward(inputs)
                # Compute loss and accuracy
                loss = self.criterion(outputs.to(self.device), targets)
                accuracy = self.get_accuracy(outputs, targets)
                # Backpropagate and compute gradients
                loss.backward()
                # Parameter update                
                self.optimizer.step()
                # Store accuracy and loss per epoch
                train_epoch_loss += loss.item()
                train_epoch_acc += accuracy.item()
                pass
                        
            with torch.no_grad():
                validation_epoch_loss = 0
                validation_epoch_acc = 0            
                self.model.eval()
                for inputs, targets in self.validation_loader:
                    # Pass inputs and targets to device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    # Get model output
                    outputs = self.forward(inputs)
                    # Compute loss and accuracy
                    loss = self.criterion(outputs.to(self.device), targets)
                    accuracy = self.get_accuracy(outputs, targets)
                    # Store accuracy and loss per epoch
                    validation_epoch_loss += loss.item()
                    validation_epoch_acc += accuracy.item() 
            
            self.loss_stats['train'].append(train_epoch_loss/len(self.train_loader))
            self.loss_stats['validation'].append(validation_epoch_loss/len(self.validation_loader))
            
            self.accuracy_stats['train'].append(train_epoch_acc/len(self.train_loader))
            self.accuracy_stats['validation'].append(validation_epoch_acc/len(self.validation_loader))
        
            epoch_loss = validation_epoch_loss / len(self.validation_loader)
            epoch_accuracy = validation_epoch_acc / len(self.validation_loader)
            print('Iteration %d, loss = %.6f \nValidation score : %.6f ' % (self.epoch, epoch_loss, epoch_accuracy))
            
            metric = epoch_accuracy
            if self.early_stopping.step(metric):
                print('Trial prunned')
                self.save_stats()
                self.trial.report(self.early_stopping.best, self.epoch)
                return self.early_stopping.best
            
        return self.early_stopping.best
            
    def get_accuracy(self, y_pred, y_true):
        # y_pred = torch.log_softmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        correct_pred = (y_pred == y_true).float()
        accuracy = correct_pred.sum() / len(correct_pred)
        return accuracy
    
    # Returns prediction for input x
    def predict(self, x):
        predictions = []
        
        test_tensors = torch.from_numpy(x).float()
        test_dataloader = DataLoader(test_tensors , self.batch_size)
        
        for batch, sample in enumerate(test_dataloader):
            outputs = self.forward(sample.to(self.device))
            y_pred = torch.argmax(outputs, dim=1)
            predictions.extend(y_pred.cpu().numpy())
            
        return predictions
    
    def save_stats(self):
        pd.DataFrame(self.accuracy_stats['train']).to_csv(self.training_score_path, index=False, header=False)
        pd.DataFrame(self.accuracy_stats['validation']).to_csv(self.validation_score_path, index=False, header=False)
        pd.DataFrame(self.loss_stats['train']).to_csv(self.training_loss_path, index=False, header=False)
        pd.DataFrame(self.loss_stats['validation']).to_csv(self.validation_loss_path, index=False, header=False)            
        pass