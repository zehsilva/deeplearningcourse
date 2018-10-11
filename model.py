import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, n_layers=2):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.n_layers = n_layers
        
        self.class_embedding = nn.Embedding(num_classes,hidden_size)
        
        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
       
    
    
    def forward(self, input_sequences, input_sequences_lengths, hidden=None):

