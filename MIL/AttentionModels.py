import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from entmax import sparsemax, entmax15, entmax_bisect

from typing import Union, Optional, Any, Tuple

from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_

import math

class MAB(nn.Module):
    """ 
    Adapted from official implementation of SetTrans: https://github.com/juho-lee/set_transformer/blob/master/modules.py"
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, dropout = 0.0, activation = 'softmax'):
        super(MAB, self).__init__()
        self.dim_Q = dim_Q
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.activation = activation
        
    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)

        if self.activation == "softmax":
            A = F.softmax(A, dim=-1)
        elif self.activation == "sparsemax":
            A = sparsemax(A, dim=-1)
        elif self.activation == "entmax15":
            A = entmax15(A, dim=-1)

        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        return O, A

class SetAttentionBlock(nn.Module):
    """ 
    Adapted from official implementation of SetTrans: https://github.com/juho-lee/set_transformer/blob/master/modules.py"
    """
    def __init__(self, d_model: int, d_hidden: int, heads: int = 1, layer_norm: bool = True, activation = 'softmax'):
        super(SetAttentionBlock, self).__init__()
        
        self.mab = MAB(d_model, d_model, d_hidden, heads, ln=layer_norm, activation = activation)

    def forward(self, X, mask = None):

        Z, _ = self.mab(X, X)
        
        return Z

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.mab.dim_Q}, '
                f'd_hidden = {self.mab.dim_V}, '
                f'heads={self.mab.num_heads}, '
                f'layer_norm={self.mab.ln0 is not None}, '
                f'activation={self.mab.activation})')


class InducedSetAttentionBlock(nn.Module):
    """ 
    Adapted from official implementation of SetTrans: https://github.com/juho-lee/set_transformer/blob/master/modules.py"
    """
    def __init__(self, d_model: int, d_hidden, num_induced_points: int, heads: int = 1, layer_norm: bool = True, activation = 'softmax'):
        super(InducedSetAttentionBlock, self).__init__()

        self.I = nn.Parameter(torch.Tensor(1, num_induced_points, d_hidden))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(d_hidden, d_model, d_hidden, heads, ln=layer_norm, activation = activation)
        self.mab1 = MAB(d_model, d_hidden, d_hidden, heads, ln=layer_norm, activation = activation)
        
    def forward(self, X, mask = None):
        H, _ = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        H, _ = self.mab1(X, H)
        return H
        
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.mab1.dim_Q}, '
                f'd_hidden = {self.mab1.dim_V}, '
                f'num_induced_points={self.I.size(1)}, '
                f'heads={self.mab1.num_heads}, '
                f'layer_norm={self.mab1.ln0 is not None}, ' 
                f'activation={self.mab1.activation})')

class PoolingByMultiheadAttention(nn.Module):
    """ 
    Adapted from official implementation of SetTrans: https://github.com/juho-lee/set_transformer/blob/master/modules.py"
    """
    def __init__(self, d_model: int, num_seed_points: int = 1, heads: int = 1, layer_norm: bool = True, activation = 'softmax'):
        super(PoolingByMultiheadAttention, self).__init__()
        
        self.S = nn.Parameter(torch.Tensor(1, 1, d_model))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(d_model, d_model, d_model, heads, ln=layer_norm, activation = activation)

    def forward(self, X, mask = None):
        
        x, A = self.mab(self.S.repeat(X.size(0), 1, 1), X) 

        x = x.nan_to_num() 
        
        return x.squeeze(1), A.squeeze(1)
        
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.S.size(2)}, '
                f'num_seed_points={self.S.size(1)}, '
                f'heads={self.mab.num_heads}, '
                f'layer_norm={self.mab.ln0 is not None}, ' 
                f'activation={self.mab.activation})')


class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    """

    def __init__(self, L = 512, D = 256, dropout = 0.0, map_prob_func = 'softmax'):
        super(Attn_Net, self).__init__()
        
        self.module = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(D, 1)
        )

        self.map_prob_func = map_prob_func
        
    def forward(self, x, mask=None):
        
        A = self.module(x) # batch_size x N x 1
        
        if mask is not None:
            A = A.masked_fill(mask, float('-inf'))

        if self.map_prob_func == 'softmax': 
            A = F.softmax(A, dim=1)  # softmax over N

        elif self.map_prob_func == 'sparsemax':
            A = sparsemax(A, dim=1)  

        elif self.map_prob_func == 'entmax':
            A = entmax15(A, dim=1)  
                
        else:
            raise ValueError(f"Unknown map probability function: {self.map_prob_func}")
                        
        # Apply mask again after attention
        if mask is not None:
            A = A.masked_fill(mask, 0.0)

        A = A.permute(0, 2, 1)  # batch_size x 1 x N
        
        # A.shape -> (Batch_size, 1, N) * (Batch_size, N, embedding_size)
        pooled_feature = torch.matmul(A,x)

        return pooled_feature.squeeze(1), A.squeeze(1) #(Batch_size, embedding_size)

class Gated_Attn_Net(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    """
    def __init__(self, L = 512, D = 256, dropout = 0.0, map_prob_func = 'softmax'):
        super(Gated_Attn_Net, self).__init__()
        
        # Initialize attention_V
        self.attention_V = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(p=dropout)
        )
        
        # Initialize attention_U
        self.attention_U = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(p=dropout)
        )
        
        self.attention_weights = nn.Linear(D, 1)
        
        self.map_prob_func = map_prob_func
    
    def forward(self, x, mask=None):

        A_V = self.attention_V(x) # batch_size x N x 1
        A_U = self.attention_U(x) # batch_size x N x 1
        A = A_V.mul(A_U) 
        A_unormalized = self.attention_weights(A)  # batch_size x N x 1

        if mask is not None:
            A_unormalized = A_unormalized.masked_fill(mask, float('-inf'))
            
        if self.map_prob_func == 'softmax': 
            A = F.softmax(A_unormalized, dim=1)  # softmax over N
        elif self.map_prob_func == 'sparsemax':
            A = sparsemax(A_unormalized, dim=1)  
        elif self.map_prob_func == 'entmax':
            A = entmax15(A_unormalized, dim=1)  
                        
        # Apply mask again after attention
        if mask is not None:
            A = A.masked_fill(mask, 0.0)
        
        A = A.permute(0, 2, 1)  # batch_size x 1 x N
            
        pooled_feature = torch.matmul(A, x)  #(Batch_size, 1, embedding_size) 

        return pooled_feature.squeeze(1), A.squeeze(1) # (Batch_size, embedding_size)

