import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

import torch
from torch import nn


class Composer(nn.Module):
    def __init__(self, input_size, hidden_size, num_tags, n_layers=2):
        super(Composer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.n_layers = n_layers
        
        self.tags_embedding = nn.Embedding(num_tags,hidden_size)
        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.notes_decoder = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.output= nn.Sigmoid()
    
    def forward(self, input_sequence, tag, hidden=None):
        if(hidden is None):
            hidden = self.tags_embedding(tag)
            hidden = torch.cat( tuple(hidden for i in range(self.n_layers))) 
        output, hidden = self.gru(self.notes_encoder(input_sequence),hidden)
        output = self.output(self.notes_decoder(output))
        return output,hidden

class ComposerLSTM_C(nn.Module):
    r""" TODO: change to LSTM with embedding entering in the initial cell state c
    """
    def __init__(self, input_size, hidden_size, num_tags, n_layers=2):
        super(ComposerLSTM_C, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.n_layers = n_layers
        
        self.tags_embedding = nn.Embedding(num_tags,hidden_size)
        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.notes_decoder = nn.Linear(in_features=hidden_size, out_features=input_size)
        
    def forward(self, input_sequence, tag, hidden=None):
        if(hidden is None):
            hidden = self.tags_embedding(tag)
            hidden = torch.cat( (hidden for i in range(self.n_layers))) 


class ComposerLSTM_H(nn.Module):
       r""" TODO: change to LSTM with embedding entering in the initial hidden state h
    """
    def __init__(self, input_size, hidden_size, num_tags, n_layers=2):
        super(ComposerLSTM_H, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_tags = num_tags
        self.n_layers = n_layers
        
        self.tags_embedding = nn.Embedding(num_tags,hidden_size)
        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.notes_decoder = nn.Linear(in_features=hidden_size, out_features=input_size)
        
    def forward(self, input_sequence, tag, hidden=None):
        if(hidden is None):
            hidden = self.tags_embedding(tag)
            hidden = torch.cat( (hidden for i in range(self.n_layers))) 


