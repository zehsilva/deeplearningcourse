from torch.autograd import Variable
import torch.utils.data as data

import torch
from torch import nn


class Composer(nn.Module):
    def __init__(self, input_size, hidden_size, num_tags, n_layers=2, usegpu=True,drop=0):
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
        self.usegpu=usegpu
        #self.output= nn.LogSigmoid()
    
    def forward(self, input_sequence, tag, hidden=None):
        if(hidden is None):
            hidden = self.tags_embedding(tag)
            hidden = torch.cat( tuple(hidden for i in range(self.n_layers))) 
        output, hidden = self.gru(self.notes_encoder(input_sequence),hidden)
        output = self.output(self.notes_decoder(output))
        return output,hidden
    
    def generate(self,tag,n,k=1):
        init = torch.randint(0,2,size=(k,1,self.input_size))
        res = init
        hidden = None
        for i in xrange(n//k):
            init,hidden = self.forward(init,tag,hidden)
            #init = torch.round(torch.exp(init))
            init = torch.round(init)
            res = torch.cat ( ( res, init ) )
        return res
    
    def generate_noround(self,tag,n,k=1,init=None):
        init = torch.randint(0,2,size=(k,1,self.input_size))
        res = init
        hidden = None
        for i in xrange(n//k):
            init,hidden = self.forward(init,tag,hidden)
            #init = torch.round(torch.exp(init))
            #init = torch.round(init)
            res = torch.cat ( ( res, init ) )
        return res
    
class Generalist(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, usegpu=True,drop=0):
        super(Generalist, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.notes_encoder = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,dropout=drop)
        self.notes_decoder = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.output= nn.Sigmoid()
        self.usegpu=usegpu
        #self.output= nn.LogSigmoid()
    
    def forward(self, input_sequence,tag,  hidden=None):
        output, hidden = self.gru(self.notes_encoder(input_sequence),hidden)
        output = self.output(self.notes_decoder(output))
        return output,hidden
    
    def generate(self,tag,n,k=1):
        init = torch.randint(0,2,size=(k,1,self.input_size))
        res = init
        hidden = None
        for i in xrange(n//k):
            init,hidden = self.forward(init,tag,hidden)
            #init = torch.round(torch.exp(init))
            init = torch.round(init)
            res = torch.cat ( ( res, init ) )
        return res
    
    def generate_noround(self,tag,n,k=1,init=None):
        init = torch.randint(0,2,size=(k,1,self.input_size))
        res = init
        hidden = None
        for i in xrange(n//k):
            init,hidden = self.forward(init,tag,hidden)
            #init = torch.round(torch.exp(init))
            #init = torch.round(init)
            res = torch.cat ( ( res, init ) )
        return res
    
class ComposerNonBinary(nn.Module):
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
        #self.output= nn.ReLU()
        #self.output= nn.LogSigmoid()
    
    def forward(self, input_sequence, tag, hidden=None):
        if(hidden is None):
            hidden = self.tags_embedding(tag)
            hidden = torch.cat( tuple(hidden for i in range(self.n_layers))) 
        output, hidden = self.gru(self.notes_encoder(input_sequence),hidden)
        #output = self.output(self.notes_decoder(output))
        return output,hidden
    
    def generate(self,tag,n,k=1):
        init = torch.randint(0,2,size=(k,1,self.input_size))
        res = init
        hidden = None
        for i in xrange(n//k):
            init,hidden = self.forward(init,tag,hidden)
            #init = torch.round(torch.exp(init))
            init = torch.round(init)
            res = torch.cat ( ( res, init ) )
        return res
    
    def generate_noround(self,tag,n,k=1):
        init = torch.randint(0,2,size=(k,1,self.input_size))
        res = init
        hidden = None
        for i in xrange(n//k):
            init,hidden = self.forward(init,tag,hidden)
            #init = torch.round(torch.exp(init))
            #init = torch.round(init)
            res = torch.cat ( ( res, init ) )
        return res

""" 
class ComposerLSTM_C(nn.Module):
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
"""

