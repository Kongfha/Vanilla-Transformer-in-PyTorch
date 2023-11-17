import torch.nn as nn

class PositionalWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_prob = 0.1):
        '''
        Args:
            d_model -> dimensionality of input and output
            d_ff -> dimensionality of inner layer 
            drop_prob -> dropout probability
        '''
        super().__init__()

        '''
        Preparing
            Linear1 as linear layer where W1 is element of R(d_model x d_ff)
            Linear2 as linear layer where W2 is element of R(d_ff x d_model)
            Relu as ReLu Function
        According to 3.3 Position-wise Feed-Forward Networks (vaswani et al. 2017).
        '''
        self.Linear1 = nn.Linear(d_model, d_ff)
        self.Linear2 = nn.Linear(d_ff, d_model)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)

    def forward(self, x):
        '''
        Args:
            x -> residual connection output from Multi-Head Attention layer
        Returns:
            x -> output from Positional-Wise-Feed-Forward layer or FFN(x)
        '''

        '''
        Performing FFN(x)
        where FFN(x) = max(0, xW1 + b1)W2 + b2
           or FFN(x) = Linear2(ReLU(Linear1(x)))
        According to 3.3 Position-wise Feed-Forward Networks (vaswani et al. 2017).
        '''
        x = self.Linear1(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.Linear2(x)

        return x