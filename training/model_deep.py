# -*- coding: utf-8 -*-
import torch

from model import Model
from tuning import Tuner

class CNNDynamic(Model):
    
    @staticmethod
    def compute_maxPool1D_out_size(input_size: int, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1) -> int:        
        if stride is None: stride = kernel_size
        return int((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    @staticmethod
    def compute_conv1d_out_size(input_size: int, kernel_size: int, padding: int = 0, dilation: int = 1, stride: int = 1) -> int:
        return int((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    
    def __init__(self, kwargs):
        super(CNNDynamic, self).__init__(kwargs)

        if 'dense_layer_sizes'  in kwargs: self.dense_layer_sizes   = kwargs['dense_layer_sizes']
        if 'dropout_rates'      in kwargs: self.dropout_rates       = kwargs['dropout_rates']
        if 'hidden_layer_sizes' in kwargs: self.hidden_layer_sizes  = kwargs['hidden_layer_sizes']
        if 'kernel_size'        in kwargs: self.kernel_size         = kwargs['kernel_size']
        if 'num_classes'        in kwargs: self.num_classes         = kwargs['num_classes']
        if 'num_features'       in kwargs: self.num_features        = kwargs['num_features']
        if 'stride'             in kwargs: self.stride              = kwargs['stride']
        
        self.in_features = 1
        self.layers = []
        self.dense = []
        
        # Generate layers
        for i in range(len(self.hidden_layer_sizes)):
            self.hidden = self.hidden_layer_sizes[i]
            self.dropout_rate = self.dropout_rates[i]
            # CNN layers
            self.layers.append(torch.nn.Conv1d(self.in_features, self.hidden, self.kernel_size, self.stride))
            # Or batchnorm here
            # self.layers.append(torch.nn.PReLU()) # Default 0.25 # Learnable by network
            self.layers.append(torch.nn.LeakyReLU(0.1, inplace=True))
            self.layers.append(torch.nn.BatchNorm1d(self.hidden))
            self.layers.append(torch.nn.Dropout(self.dropout_rate))
            self.layers.append(torch.nn.MaxPool1d(self.kernel_size, self.stride))
            # Calculate conv and pool out sizes
            self.conv_out_size = self.compute_conv1d_out_size(self.num_features, self.kernel_size)
            self.pool_out_size = self.compute_maxPool1D_out_size(self.conv_out_size, self.kernel_size, self.stride)
            # Update in_features
            self.in_features = self.hidden
            self.num_features = self.pool_out_size
            pass
        
        # Dense : 3 Layer
        self.dense.append(torch.nn.Linear(self.pool_out_size * self.hidden, self.dense_layer_sizes[0]))
        self.dense.append(torch.nn.Linear(self.dense_layer_sizes[0], self.dense_layer_sizes[1]))
        self.dense.append(torch.nn.Linear(self.dense_layer_sizes[1], self.num_classes))
        
        #
        self.model = torch.nn.Sequential(*self.layers)
        self.dense = torch.nn.Sequential(*self.dense)
        pass
    
    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.model(x) 
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

class RNN(Model):
    
    def __init__(self, kwargs):
        super(RNN, self).__init__(kwargs)

        if 'device'         in kwargs: self.device          = kwargs['device']
        if 'dropout_rate'   in kwargs: self.dropout_rate    = kwargs['dropout_rate']
        if 'hidden_size'    in kwargs: self.hidden_size     = kwargs['hidden_size']
        if 'num_layers'     in kwargs: self.num_layers      = kwargs['num_layers']
        if 'num_classes'    in kwargs: self.num_classes     = kwargs['num_classes']
        if 'num_features'   in kwargs: self.num_features    = kwargs['num_features']
        
        self.model = torch.nn.RNN(1, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_rate)
        self.fc = torch.nn.Linear(self.hidden_size, self.num_classes)
        self.relu = torch.nn.ReLU()
        pass

    def forward(self, x):
        x = x.view(-1, self.num_features, 1)
        hidden = self.init_hidden(x.shape[0])
        x, hidden_state = self.model(x, hidden)
        x = self.fc(self.relu(x[:,-1,:]))
        return x
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return hidden


class LSTM(Model):
    
    def __init__(self, kwargs):
        super(LSTM, self).__init__(kwargs)

        if 'device'         in kwargs: self.device          = kwargs['device']
        if 'dropout_rate'   in kwargs: self.dropout_rate    = kwargs['dropout_rate']
        if 'hidden_size'    in kwargs: self.hidden_size     = kwargs['hidden_size']
        if 'num_layers'     in kwargs: self.num_layers      = kwargs['num_layers']
        if 'num_classes'    in kwargs: self.num_classes     = kwargs['num_classes']
        if 'num_features'   in kwargs: self.num_features    = kwargs['num_features']
        
        self.model = torch.nn.LSTM(1, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_rate)
        self.fc = torch.nn.Linear(self.hidden_size, self.num_classes)
        self.relu = torch.nn.ReLU()    
        pass

    def forward(self, x):
        x = x.view(-1, self.num_features, 1)
        hidden = self.init_hidden(x.shape[0])
        x, hidden_state = self.model(x, hidden)
        x = self.fc(self.relu(x[:,-1,:]))
        return x
    
    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                  torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
        return hidden

class GRU(Model):
    
    def __init__(self, kwargs):
        super(GRU, self).__init__(kwargs)
        
        if 'device'         in kwargs: self.device          = kwargs['device']
        if 'dropout_rate'   in kwargs: self.dropout_rate    = kwargs['dropout_rate']
        if 'hidden_size'    in kwargs: self.hidden_size     = kwargs['hidden_size']
        if 'num_layers'     in kwargs: self.num_layers      = kwargs['num_layers']
        if 'num_classes'    in kwargs: self.num_classes     = kwargs['num_classes']
        if 'num_features'   in kwargs: self.num_features    = kwargs['num_features']
        
        self.model = torch.nn.GRU(1, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_rate)
        self.fc = torch.nn.Linear(self.hidden_size, self.num_classes)
        self.relu = torch.nn.ReLU()    
        pass

    def forward(self, x):
        x = x.view(-1, self.num_features, 1)
        hidden = self.init_hidden(x.shape[0])
        x, hidden_state = self.model(x, hidden)
        # x = self.fc(self.relu(x[:,-1,:]))
        x = self.fc(x[:,-1,:])
        return x
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return hidden
    
    

# class HybridModel(Model):
    
#     def __init__(self, trial, num_features, num_classes):
#         cnn = CNNDynamic(trial, num_features, num_classes)
#         lstm = LSTM(trial, input_size, output_size)
#         pass
    
#     def forward(self, encodings, embeddings):
#         out_cnn = cnn.model(encodings)
#         out_lstm, _ = lstm.model(embeddings)

# fc of cnn
# fc of lstm
# concatenate both tensors to num_classes
        
            
#         pass
    