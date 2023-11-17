import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim = 512, n_heads = 8):
        '''
        Args:
            embedding_dim -> the number of dimensions in word embedding vector (dmodel)
            n_heads -> number of heads for self-attention (h)
        '''
        super().__init__()

        '''
        Preparing h, dk, and dv values and linear transformation for q, k, and v matrix. 
        Thus, Wqi, Wki project from dmodel to dk (linear transformation weight for q and k in headi)
                Wvi projects from dmodel to dv (linear transformation weight for v in headi)
                Wo projects from h*dv to dmodel (linear transformation weight for output)
        where dk = dv = dmodel/h, 
        according to 3.2.2 Multi-Head Attention (vaswani et al. 2017).
        '''
        self.embed_dim = embedding_dim # value of dmodel
        self.n_heads = n_heads # value of h
        self.single_head_dim = self.embed_dim // self.n_heads # value of dk,dv

        '''
        Preparing Linear transformation for q,k,v, and output.
        For Wq Wk Wv will linearly project q,k,v from dmodel into the size of dmodel.
        Though, the q,k,v will be split into the size of dk = dv afterward.
        This approach avoids the need of implementing h matrices of Wqi, Wki, and Wvi independently.
        '''
        self.Wq = nn.Linear(self.embed_dim, self.embed_dim, bias = False)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim, bias = False)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim, bias = False)
        self.Wo = nn.Linear(self.embed_dim, self.embed_dim, bias = False)

    def forward(self, key, query, value, mask = None):
        '''
        Args:       
            key -> key vector
            query -> query vector
            value -> value vector
            mask -> mask for decoder
        Returns:
            output -> output vector from multihead attention
        '''
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # In decoder the seq_length may not be the same with encoder
        seq_length_query = query.size(1) 

        # Linear projecting k, q and v.
        k = self.Wk(key)  
        q = self.Wq(query)
        v = self.Wv(value)

        # Split linear projected matrices to (batch_size, seq_length, n_heads, single_head_dim).
        k = k.view(batch_size, seq_length, self.n_heads, self.single_head_dim) 
        q = q.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        v = v.view(batch_size, seq_length, self.n_heads, self.single_head_dim) 

        # Adjust tensors for torch.matmul operation
        # from (batch_size, seq_length, n_heads, single_head_dim)
        # to (batch_size, n_heads, seq_length, single_head_dim)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        '''
        Perfroming Attention(Q, K, V)
        according to 3.2.1 Scaled Dot-Product Attention (vaswani et al. 2017).
        '''
        # Get transposed k matrix (Kt) (batch_size, n_heads, single_head_dim, seq_length)
        k_t = k.transpose(-1,-2)

        qkt_product = torch.matmul(q,k_t) # (batch_size, n_heads, q_seq_length, k_seq_length)
        
        # Masking self-attention matrix for decoder
        if mask is not None:
            qkt_product = qkt_product.masked_fill(mask==0, float("-1e20"))

        # Scale the self-attention matrix by dividing by sqrt(dmodel)
        qkt_product = qkt_product / math.sqrt(self.embed_dim)

        # Apply softmax for a word in q with all words in key
        scores = F.softmax(qkt_product, dim = 3)

        # Multiply the scores to value matrix
        scores = torch.matmul(scores, v)

        # Concat every head
        scores = scores.transpose(1,2).contiguous() # adjust tensor for using tensor.veiw
        concat = scores.view(batch_size, seq_length_query, self.n_heads * self.single_head_dim)

        # Linear transform the concat matrix using Wo to get output
        output = self.Wo(concat)

        return output       