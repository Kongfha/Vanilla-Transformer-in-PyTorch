import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        '''
        Args:
            vocab_size -> the number of vocabulary within the model
            embedding_dim -> the number of dimension in word embedding vector 
        '''
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        '''
        Args:
            x -> input vector
        Returns:
            output -> word embedded vector 
        '''
        output = self.embed(x)
        return output