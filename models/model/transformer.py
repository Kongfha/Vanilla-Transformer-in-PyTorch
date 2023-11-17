from torch import nn
import torch.nn.functional as F
import torch

from models.model.encoder import Encoder
from models.model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, max_length, d_model, n_heads,
                 d_ff, n_layers, drop_prob, device):
        '''
        Args:
            src_pad_idx -> Encoder padding token id
            trg_pad_idx -> Decoder padding token id
            enc_voc_size -> Encoder vocabulary size
            dec_voc_size -> Decoder vocabulary size
            max_length -> Transformer max sequence length
            d_model -> embedding dimension
            n_heads -> number of attention heads
            d_ff -> hidden layer dimension for feed forward network
            n_layers -> number of encoder and decoder layers
            drop_prob -> dropout probability
            device -> hardware device setting
        '''
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.encoder = Encoder(enc_voc_size, max_length, d_model, d_ff,
                               n_heads, n_layers, drop_prob, device)
        
        self.decoder = Decoder(dec_voc_size, max_length, d_model, d_ff,
                               n_heads, n_layers, drop_prob, device)
        
    def forward(self, src, trg, do_softmax = False):
        '''
        Args: 
            src -> Source token tensors for Encoder
            trg -> Target token tensors for Decoder
        Returns:
            out -> Transformer output word probability ()
        '''
        # Get all masks for each Multi-Head Attention Layer
        enc_pad_mask = self.get_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)

        masked_mha_mask = self.get_look_ahead_mask(trg) * \
                          self.get_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) 
        
        enc_dec_pad_mask = self.get_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc_out = self.encoder(src, enc_pad_mask)
        out = self.decoder(trg, enc_out, masked_mha_mask, enc_dec_pad_mask) # logits

        if do_softmax:
            out = F.softmax(out, dim=2)

        return out

    
    def get_pad_mask(self, q_tokens_tensor, k_tokens_tensor, q_pad_idx, k_pad_idx):
        '''
        Args:
            q_tokens_tensor -> tokens tensor for query 
            k_tokens_tensor -> tokens tensor for key
            q_pad_idx -> padding token id for query tokens tensor
            k_pad_idx -> padding token id for key tokens tensor
        Return:
            mask -> Padding mask where only the cell that has non-padding id 
                    from both key and query gets marked as True
        '''

        # Get sequence length for query and key
        q_length = q_tokens_tensor.size(1)
        k_length = k_tokens_tensor.size(1)

        # Map padding ids as False and others as True
        k = k_tokens_tensor.ne(k_pad_idx)
        # modify the tensor shape to the shape of qkt (product of q and transpose k)
        k = k.unsqueeze(1).unsqueeze(2) # batch_size x 1 x 1 x k_length
        k = k.repeat(1, 1, q_length, 1) # batch_size x 1 x q_length x k_length

        # Map padding ids as False and others as True
        q = q_tokens_tensor.ne(q_pad_idx)
        # modify the tensor shape to the shape of qkt (product of q and transpose k)
        q = q.unsqueeze(1).unsqueeze(3) # batch_size x 1 x q_length x 1
        q = q.repeat(1, 1, 1, k_length) # batch_size x 1 x q_length x k_length

        # Make a mask where only the cell that has non-padding id from both key and query gets True
        mask = k * q

        return mask
    
    def get_look_ahead_mask(self, trg):
        '''
        Args:
            trg -> Decoder inputs tensors
        Returns:
            mask -> 2D mask tensor with lower triangular value as True and higher triangular as False
        '''
        seq_length = trg.size(1)
        mask = torch.tril(torch.ones(seq_length,seq_length).type(torch.BoolTensor)).to(self.device)
        
        return mask