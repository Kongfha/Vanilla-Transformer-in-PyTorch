from torch import nn

from models.layers.word_embedding import Embedding
from models.layers.pos_embedding import PositionalEmbedding
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.positional_wise_feed_forward import PositionalWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff_hidden, n_heads, drop_prob):
        '''
        Args:
            d_model -> embedding dimension (d_model)
            d_ff_hidden -> feed forward hidden layer dimension (d_ff)
            n_heads -> number of attention heads
            drop_prob -> dropout probability
        '''
        super().__init__()

        # Multi-Head Attention section's layers
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        # Feed Forward section's layers
        self.ffn = PositionalWiseFeedForward(d_model, d_ff_hidden, drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, s_mask):
        '''
        Args:
            x -> Encoder layer's input 
            s_mask -> Self-Attention mask (padding-mask)
        Returns:
            x -> Encoder layer's output
        '''
        
        '''
        Performing Encoding Layer,
        according to 3.1 Encoder and Decoder Stacks (vaswani et al. 2017).
        '''
        # ---- Multi-Head Attention ----
        # Copy the input for residual conneciton (Multi-Head Attention section)
        _x = x

        # Perform Multi-Head Attention
        x = self.attention(key = x, query = x, value = x, mask = s_mask)
        x = self.dropout1(x)
    
        # Perform add & norm
        x = self.norm1(x + _x)


        # ---- Feed Forward ----
        # Copy the input for residual conneciton (Feed Forward section)
        _x = x

        # Perform Positional-Wise-Feed-Forward
        x = self.ffn(x) 
        x = self.dropout2(x)
        
        # Perform add & norm
        x = self.norm2(x + _x)

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_length, d_model, d_ff, n_heads, n_layers, drop_prob, device):
        '''
        Args:
            vocab_size -> vocabulary size  
            max_length -> maximum length of the input tensor
            d_model -> embedding dimension
            d_ff -> hidden layer dimension for feed forward network
            n_heads -> number of attention heads
            n_layers -> number of encoder layers
            drop_prob -> dropout probability
            device -> hardware device setting
        '''
        super().__init__()

        # Word and Possitional Embedding Layer
        self.word_embedding = Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(max_length, d_model, device)

        # Encoder Layer Stacks
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,
                                                          d_ff,
                                                          n_heads,
                                                          drop_prob)
                                                for _ in range(n_layers)])
        
    def forward(self, x, s_mask):
        '''
        Args:
            x -> input
            s_mask -> Self-Attention mask (Padding-mask)
        Returns:
            x -> Encoder block output tensor
        '''

        '''
        Performing Encoder block
        according to Figure 1: The Transformer - model architecture. (vaswani et al. 2017).
        '''

        # Perform input embedding
        x = self.word_embedding(x)
        x = self.pos_embedding(x)

        # Perform all Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, s_mask)

        return x