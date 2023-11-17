from torch import nn

from models.layers.word_embedding import Embedding
from models.layers.pos_embedding import PositionalEmbedding
from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.positional_wise_feed_forward import PositionalWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, drop_prob):
        '''
        Args:
            d_model -> embedding dimension (d_model)
            d_ff -> hidden layer dimension (d_ff)
            n_heads -> number of Attention heads
            drop_prob -> dropout probability
        '''
        super().__init__()

        self.attention1 = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.attention2 = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionalWiseFeedForward(d_model, d_ff, drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, x, enc_output, dec_mask, enc_dec_mask):
        '''
        Args:
            x -> decoder input (target)
            enc_output -> output from encoder
            dec_mask -> no-peek mask for first Multi-Head attention layer (Masked Multi-Head Attention)
            enc_dec_mask -> padding mask for second Multi-Head attention layer (Encoder-Decoder Attention)
        Returns:
            x -> decoder output
        '''

        '''
        Performing Encoding Layer,
        according to 3.1 Encoder and Decoder Stacks (vaswani et al. 2017).
        '''
        # ---- Masked Multi-Head Attention ----
        # Copy the decoder input for residual conneciton (Masked Multi-Head Attention section)
        _x = x

        # Perform Masked Multi-Head Attention
        x = self.attention1(key = x, query = x, value = x, mask = dec_mask)
        x = self.dropout1(x)

        # Perform add&norm
        x = self.norm1(x + _x)
        
        # ---- Encoder-Decoder Attention ----
        if enc_output is not None:
            # Copy the prior layer output for risidual connection (Encoder-Decoder Attention section)
            _x = x

            # Perform Encoder-Decoder Attention
            x = self.attention2(key = enc_output, query = x, value = enc_output, mask = enc_dec_mask)
            x = self.dropout2(x)

            # Peroform add&norm
            x = self.norm2(x + _x)
        
        # ---- Position-Wise Feed-Forward Network ----
        # Copy the prior layer output for risidual connection (Position-Wise Feed-Forward Network section)
        _x = x

        # Perform Position-Wise Feed-Forward Network 
        x = self.ffn(x)
        x = self.dropout3(x)

        # Perform add&norm
        x = self.norm2(x +_x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_length, d_model, d_ff, n_heads, n_layers, drop_prob, device):
        '''
        Args:
            vocab_size -> vocabulary size  
            max_length -> maximum length of the input tensor
            d_model -> embedding dimension
            d_ff -> hidden layer dimension for feed forward network
            n_heads -> number of attention heads
            n_layers -> number of decoder layers
            drop_prob -> dropout probability
            device -> hardware device setting
        '''
        super().__init__()

        # Word and Possitional Embedding Layer
        self.word_embedding = Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEmbedding(max_length, d_model, device)

        # Decoder Stack
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,
                                                          d_ff,
                                                          n_heads,
                                                          drop_prob)
                                                for _ in range(n_layers)])
        
        # Final Linear Layer
        self.final_linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_output, dec_mask, enc_dec_mask):
        '''
        Args:
            x -> decoder input (target)
            enc_output -> output from encoder
            dec_mask -> look-ahead mask for first Multi-Head attention layer (Masked Multi-Head Attention)
            enc_dec_mask -> padding mask for second Multi-Head attention layer (Encoder-Decoder Attention)
        Returns:
            x -> Output logits
        '''

        '''
        Performing Encoder block
        according to Figure 1: The Transformer - model architecture. (vaswani et al. 2017).
        '''
        
        # Perform input embedding
        x = self.word_embedding(x)
        x = self.pos_embedding(x)

        # Perform decoder layers
        for layer in self.decoder_layers:
            x = layer(x, enc_output, dec_mask, enc_dec_mask)

        # Perform LM Head
        x = self.final_linear(x) # (batch size, seq_length, vocab size)

        return x