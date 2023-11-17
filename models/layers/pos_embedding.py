import math
import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, embedding_dim, device):
        '''
        Args:
            max_length -> the max length of the input matrix (embedded inputs)
            embedding_dim -> the number of dimension in word embedding vector
            device -> hardware device setting
        '''
        super().__init__()
        self.embed_dim = embedding_dim

        '''
        Creating Positional Embedding Matrix (tensors),
        according to Sinusoidal Positional Embedding (vaswani et al. 2017).
        '''

        pe = torch.zeros(max_length,self.embed_dim, device = device)

        for pos in range(max_length):
            for i in range(0, self.embed_dim, 2):
                pe[pos,i] = math.sin(pos  / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos,i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / self.embed_dim))) 
                # embedding_dim should be even number to prevent out of range error
        
        pe = pe.unsqueeze(0) # add batch dimension
        self.register_buffer('pe', pe) # to save pe tensor in state_dict even it will not be trained
    
    def forward(self, x):
        '''
        Args:
            x -> input matrix (word embedded inputs tensors)
        Returns:
            x -> positional embedded input tensors
        '''

        # Scale the input matrix (making it larger to maintain the magnitude of the value in further steps)
        x = x * math.sqrt(self.embed_dim)

        # Get the input length from second dim of input tensor (batch dim included)
        seq_len = x.size(1)

        # Add positional encoder to input tensors 
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)

        return x