# internal imports 
from .AttentionModels import *

# external imports 
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import math

from typing import Union, Optional, Any, Tuple

from torch.nn.parameter import Parameter

class head(nn.Module):    
    """
    A classification head. 
    Also supports computing scale-specific weights. 

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output classes.
        sigmoid_func (bool): Whether to apply sigmoid to the output.
        dropout (float): Dropout probability.
    """
    def __init__(self, in_features, out_features, sigmoid_func, dropout=0.0):
        super(head, self).__init__()
        
        self.drop = nn.Dropout(p=dropout, inplace=True)
        
        if sigmoid_func:
            self.head_classifier = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features), 
                nn.Sigmoid())
        else: 
            self.head_classifier = nn.Linear(in_features=in_features, out_features=out_features)

    def compute_scale_weights(self, x, num_scales, feat_dim):
        """
        Computes softmax-normalized weights for each scale based on feature importance in classification.

        Args:
            x (Tensor): Input features of shape (B, num_scales * feat_dim).
            num_scales (int): Number of scales.
            feat_dim (int): Dimension of each scale feature.

        Returns:
            normalized_weights (Tensor): Softmax weights across scales (B, 1, num_scales).
        """

        # Access weights from the linear layer
        W_c = self.head_classifier[0].weight if isinstance(self.head_classifier, nn.Sequential) else self.head_classifier.weight

        # Compute scale-specific weights
        scale_weights = []
        for i in range(num_scales):
            start = i * feat_dim
            end = start + feat_dim

            x_scale = x[:, start:end] # Shape: (B, feat_dim)
            W_s = W_c[:, start:end] # Shape: (1, feat_dim)

            # Compute dot product for the scale
            scale_weight = torch.matmul(x_scale, W_s.T)  # Shape: (Batch_size, 1)

            scale_weights.append(scale_weight)

        # Stack the scale weights along a new dimension
        scale_weights = torch.stack(scale_weights, dim=-1)  # Shape: (Batch_size, 1, num_scales)
    
        # Apply softmax across the scales
        normalized_weights = F.softmax(scale_weights, dim=-1)  # Softmax over num_scales
        
        return normalized_weights

    def forward(self, x, num_scales=None, feat_dim=None, is_training = True):
        """ 
        Forward pass for the classification head.

        Args:
            x (Tensor): Input feature tensor of shape (B, in_features).
            num_scales (int, optional): Number of scales (used only in eval mode).
            feat_dim (int, optional): Feature dim per scale (used only in eval mode).
            is_training (bool): Indicates training or evaluation mode.

        Returns:
            Tensor: Class logits or probabilities.
            Optional[Tensor]: Scale weights (only in eval mode with multi-scale input).
        
        """
        x = self.drop(x)
        
        if not is_training and num_scales is not None and feat_dim is not None:  

            scale_weights = self.compute_scale_weights(x, num_scales, feat_dim)

            x = self.head_classifier(x)
            
            return x, scale_weights
        
        else: 
            x = self.head_classifier(x)
            
            return x

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, x, mask=None):
        pooled_feature,_ = torch.max(x, dim=1)
            
        return pooled_feature
                        
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, x, mask=None):

        if mask is None:
            pooled_feature = torch.mean(x, dim=1)
        else:
            pooled_feature = torch.nanmean(x.masked_fill(mask, float('nan')), dim=1)
                
        return pooled_feature
        

class ConcatAggregator(nn.Module):
    """
    Concatenates features across multiple scales
    """
    def __init__(self):
        super(ConcatAggregator, self).__init__()
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, num_scales, feat_dim).

        Returns:
            Tensor: Concatenated feature vector of shape (B, num_scales * feat_dim).
        """
    
        batch_size, num_scales, feature_dim = x.shape 

        x = x.view(batch_size, num_scales * feature_dim)
        
        return x
        
class MIL(nn.Module):
    
    def __init__(self,
                is_training: bool = True, 
                multi_scale_model: str = None,
                inst_encoder: nn.Module = None, 
                embedding_size: int = 512, 
                sigmoid_func: bool = True,
                num_classes: int = 1,
                drop_classhead: float = 0.0,
                map_prob_func:str = 'softmax',
                # MIL encoder args
                fcl_encoder_dim: int = 256,
                fcl_dropout: float = 0.0, 
                type_mil_encoder: str = 'sab', 
                num_encoder_blocks: int = 2,
                sab_num_heads: int = 4, 
                isab_num_heads: int = 4, 
                # MIL aggregator args 
                pooling_type: str = "mean",
                fcl_attention_dim = 128,
                drop_attention_pool: float = 0.0, 
                pma_num_heads: int = 1,  
                drop_mha: float = 0.0, 
                trans_layer_norm: bool = False) -> None:
    
        super().__init__()
        
        self.num_classes = num_classes
        self.sigmoid_func = sigmoid_func
        self.drop_classhead = drop_classhead
        self.map_prob_func = map_prob_func
    
        self.embedding_size = embedding_size
        self.pooling_type = pooling_type.lower()
        self.type_mil_encoder = type_mil_encoder.lower()
        self.fcl_dropout = fcl_dropout 
        
        # attention parameters
        self.fcl_attention_dim = fcl_attention_dim
        self.drop_attention_pool = drop_attention_pool
        
        # Transformer parameters
        self.fcl_encoder_dim = fcl_encoder_dim
        self.num_encoder_blocks = num_encoder_blocks
        self.sab_num_heads = sab_num_heads
        self.isab_num_heads = isab_num_heads
        self.pma_num_heads = pma_num_heads 
        self.drop_mha = drop_mha
        self.trans_layer_norm = trans_layer_norm

        self.is_training = is_training

        self.multi_scale_model = multi_scale_model
        
        self.inst_encoder = inst_encoder
        
        if not multi_scale_model: 
            self.encoder = self.MILEncoder(embedding_size, fcl_encoder_dim, scale = 0)
            self.aggregator = self.MILAggregator(fcl_encoder_dim)
            self.classifier = head(fcl_encoder_dim, num_classes, sigmoid_func, drop_classhead)

    def MILEncoder(self, dim_in, dim_hidden, scale = None, type_encoder = None) -> None:
        """
        Builds MIL encoder module (MLP, SAB, or ISAB).
        """
        
        type_encoder = self.type_mil_encoder if type_encoder is None else type_encoder 
        
        if type_encoder == 'mlp':
            encoder = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(dim_in, dim_hidden),
                        nn.ReLU(), 
                        nn.Dropout(p=self.fcl_dropout) 
                    )                     
                ])

        elif type_encoder == 'sab':
            encoder = nn.ModuleList([
                SetAttentionBlock(d_model= dim_in if i == 0 else dim_hidden, 
                                  d_hidden = dim_hidden, 
                                  heads=self.sab_num_heads, 
                                  layer_norm=self.trans_layer_norm, 
                                  activation = self.map_prob_func)
                for i in range(self.num_encoder_blocks)
            ])
        
            
        elif type_encoder == 'isab':
            
            num_induced_points = math.ceil(10*math.log10((self.num_inst[scale]))) if self.num_inst[scale] > 10 else self.num_inst[scale]
            
            encoder = nn.ModuleList([
                InducedSetAttentionBlock(d_model= dim_in if i == 0 else dim_hidden, 
                                         d_hidden = dim_hidden, 
                                         num_induced_points = num_induced_points,
                                         heads=self.isab_num_heads, 
                                         layer_norm=self.trans_layer_norm, 
                                         activation = self.map_prob_func)
                for i in range(self.num_encoder_blocks)
            ])

        return encoder
        
    def MILAggregator(self, dim_in, type_pooling = None) -> nn.Module:
        """
        Builds the MIL aggregator module.
        """
        
        type_pooling = self.pooling_type if type_pooling is None else type_pooling
        
        if type_pooling == "max":
            aggregator = MaxPooling()
        elif type_pooling == "mean":
            aggregator = MeanPooling()
        elif type_pooling == "attention":
            aggregator = Attn_Net(dim_in, self.fcl_attention_dim, self.drop_attention_pool, self.map_prob_func)
        elif type_pooling == "gated-attention":
            aggregator = Gated_Attn_Net(dim_in, self.fcl_attention_dim, self.drop_attention_pool, self.map_prob_func)
        elif type_pooling == "pma":
            aggregator = PoolingByMultiheadAttention(d_model=dim_in, 
                                                     heads=self.pma_num_heads, 
                                                     layer_norm=self.trans_layer_norm, 
                                                     activation = self.map_prob_func)
        else:
            raise ValueError(f"Invalid pooling_type: {self.pooling_type}. Must be 'max', 'mean', 'attention', 'gated-attention', 'pma'.")
        
        return aggregator

class EmbeddingMIL(MIL):
    
    def __init__(self, mil_type: str = "embedding", num_inst = None, mil_args = None) -> None:
        
        self.mil_type = mil_type.lower()
        self.num_inst = num_inst 
        
        self.patch_scores = None 
        
        super().__init__(**mil_args)
        
    def save_patch_scores(self, A):
        """Save patch-level scores attention weights."""
        self.patch_scores = A
        
    def get_patch_scores(self):
        """Retrieve saved patch-level attention scores."""
        return self.patch_scores
        
    def forward(self, x, bag_mask=None):
        """
        Forward pass of the EmbeddingMIL model.

        Args:
            x (Tensor): Shape [batch_size, num_patches, C, H, W]
            bag_mask (Tensor, optional): Optional mask for attention 

        Returns:
            Tensor: Bag-level predictions of shape [batch_size] 
        """
        
        if self.inst_encoder is not None: # online feature extraction 
            batch_size, num_patches, C, H, W = x.size()

            # Flatten the batch and patches dimensions for feature extraction
            x = x.view(-1, C, H, W)  # New size: [batch_size * num_patches, C, H, W]
        
            x = self.inst_encoder(x)  # Extract features
        
            # Reshape to [batch_size, num_patches, feature_size]
            x = x.view(batch_size, num_patches, -1) 

        for block_encoder in self.encoder:
            if isinstance(block_encoder, (SetAttentionBlock, InducedSetAttentionBlock)):
                x = block_encoder(x, bag_mask)  # Pass bag_mask for attention blocks
            else:
                x = block_encoder(x)  

        # Apply a MIL aggregator to obtain the bag representation: (Batch_size, N, embedding_size) -> (Batch_size, embedding_size)
        if self.pooling_type in ["attention", "gated-attention", "pma"]:
            x, A = self.aggregator(x, bag_mask)
            self.save_patch_scores(A)
        else: 
            x = self.aggregator(x, bag_mask)
        
        # Apply a MLP to obtain the bag label: (Batch_size, embedding_size) -> (Batch_size, 1)
        x = self.classifier(x)

        return x.squeeze(1) #(Batch_size)

class PyramidalMILmodel(MIL):
    def __init__(self, 
                 type_scale_aggregator, 
                 deep_supervision, 
                 scales, 
                 num_inst = None, 
                 mil_args = None) -> None:
        
        self.mil_type = 'embedding'
        self.num_inst = num_inst 
        
        super().__init__(**mil_args)

        self.type_scale_aggregator = type_scale_aggregator
        self.deep_supervision = deep_supervision 
        self.scales = scales

        # scale-specific encoders and aggregators
        self.side_inst_aggregator = nn.ModuleDict({
            'encoders': nn.ModuleDict({f'encoder_{scale}': self.MILEncoder(self.embedding_size, self.fcl_encoder_dim, scale_idx) for scale_idx, scale in enumerate(self.scales)}),
            'aggregators': nn.ModuleDict({f'aggregator_{scale}': self.MILAggregator(self.fcl_encoder_dim) for scale in self.scales})
        })
        
        if self.type_scale_aggregator in ['concatenation', 'gated-attention']: 
            
            if deep_supervision:
                self.side_classifiers = nn.ModuleDict({f'classifier_{scale}': head(self.fcl_encoder_dim, self.num_classes, self.sigmoid_func, self.drop_classhead) for scale in self.scales})
                
            self.scale_aggregator = self.ScaleAggregator(type_scale_aggregator, dim_in = self.fcl_encoder_dim)
            
            # final classifier head at the multi-scale aggregation level 
            if type_scale_aggregator == 'concatenation':
                self.classifier = head(self.fcl_encoder_dim*len(self.scales), self.num_classes, self.sigmoid_func, self.drop_classhead)
            
            else: 
                self.classifier = head(self.fcl_encoder_dim, self.num_classes, self.sigmoid_func, self.drop_classhead) 

        elif self.type_scale_aggregator in ['max_p', 'mean_p']:
            
            self.side_classifiers = nn.ModuleDict({f'classifier_{scale}': head(self.fcl_encoder_dim, self.num_classes, self.sigmoid_func, self.drop_classhead) for scale in self.scales})
  
        self.patch_scores = {}
        self.scale_scores = None 

        self.bag_embed = {}
        self.inst_embed = {}

    def ScaleAggregator(self, type_scale_aggregator, dim_in) -> nn.Module:
        """
        Multi-scale aggregator module.
        """
        if type_scale_aggregator in ['concatenation']: 
            scale_aggregator = ConcatAggregator()
        elif type_scale_aggregator == 'attention':
            scale_aggregator =  Attn_Net(dim_in, self.fcl_attention_dim, self.drop_attention_pool, self.map_prob_func)
        elif type_scale_aggregator == 'gated-attention':
            scale_aggregator =  Gated_Attn_Net(dim_in, self.fcl_attention_dim, self.drop_attention_pool, self.map_prob_func)
        else:
            raise ValueError(f"Unknown type_scale_aggregator: {type_scale_aggregator}")
            
        return scale_aggregator

    def save_patch_scores(self, A, s):
        self.patch_scores[s] = A
     
    def save_scale_scores(self, A):
        self.scale_scores = A
        
    def get_patch_scores(self):
        return self.patch_scores

    def get_scale_scores(self):
        return self.scale_scores
        
    def forward(self, x, bag_mask=None):
        
        # --- Instance Feature Extraction --- 
        if self.inst_encoder is not None: 
            if self.multi_scale_model in ['fpn', 'backbone_pyramid']: 
                
                if isinstance(x, list): # only top-down FPN online refinement over bottom-up backbone feature maps extracted offline 
                    
                    batch_size, num_patches, _, _, _ = x[0].size()
                    x = [tensor.view(-1, tensor.size(2), tensor.size(3), tensor.size(4)) for tensor in x] 

                else: # online feature extraction 
                    
                    # x should be a tensor of size [batch_size, num_patches, C, H, W]
                    batch_size, num_patches, C, H, W = x.size()
                
                    # Flatten the batch and patches dimensions for feature extraction
                    x = x.view(-1, C, H, W)  # New size: [batch_size * num_patches, C, H, W]
                    
                x = self.inst_encoder(x)  # Extract features

                # Reorganize feature maps into pyramid structure
                x_pyramid = OrderedDict() 
                for key, fmap in x.items(): 
    
                    _, channels, height, width = fmap.shape 
                            
                    # Transform input: (Batch_size * num_patches, embedding_size, H, W) -> (batch_size, num_patches, embedding_size, H, W)
                    fmap = fmap.view(batch_size, num_patches, channels, height, width) 
                    
                    fmap = fmap.permute(0, 1, 3, 4, 2) # shape (batch_size, num_patches, H, W, embedding_size)
                    fmap = fmap.reshape(fmap.size(0), -1, fmap.size(4)) # shape (batch_size, num_patches*H*W, embedding_size)
                                
                    x_pyramid[key] = fmap 
                
            elif self.multi_scale_model == 'msp': # Multi-scale patches (MSP)
                    
                x_pyramid = OrderedDict() 
                    
                for idx, scale in enumerate(self.scales): 
                    x_scale = x[scale]
                        
                    batch_size, num_patches, C, H, W = x_scale.size()
            
                    # Flatten the batch and patches dimensions for feature extraction
                    x_scale = x_scale.view(-1, C, H, W)  # New size: [batch_size * num_patches, C, H, W]
                
                    x_scale = self.inst_encoder(x_scale)  # Extract features
            
                    # Reshape output to [batch_size, num_patches, -1]
                    x_pyramid[f'feat_{idx}'] = x_scale.view(batch_size, num_patches, -1)
                
        else:
            if self.multi_scale_model == 'msp': 
                x_pyramid = OrderedDict({
                    f'feat_{idx}': x[scale] for idx, scale in enumerate(self.scales)
                })
        
        if self.type_scale_aggregator in ['concatenation', 'gated-attention']: 
            scale_outputs = []

            if self.deep_supervision: 
                deep_spv_outputs = []

        elif self.type_scale_aggregator in ['max_p', 'mean_p']:
            deep_spv_outputs = [] 
            
        for scale in self.scales: 
                
            x_patches = x_pyramid[f'feat_{self.scales.index(scale)}'] # Get the correct scale feature map

            # Pass through scale-specific MIL encoder
            for block_encoder in self.side_inst_aggregator['encoders'][f'encoder_{scale}']:
                if isinstance(block_encoder, (SetAttentionBlock, InducedSetAttentionBlock)):
                    x_patches = block_encoder(x_patches, bag_mask)
                else: 
                    x_patches = block_encoder(x_patches)
                                        
            # Apply the scale-specific aggregator to the encoded output to obtain the bag representation: 
            # (Batch_size, N, embedding_size) -> (Batch_size, embedding_size)
            if self.pooling_type in ["attention", "gated-attention", "pma"]:
                x_patches, A = self.side_inst_aggregator['aggregators'][f'aggregator_{scale}'](x_patches, bag_mask)
                self.save_patch_scores(A, scale)
            else:
                x_patches = self.side_inst_aggregator['aggregators'][f'aggregator_{scale}'](x_patches, bag_mask)

            # Save scale embeddings and deep supervision output
            if self.type_scale_aggregator in ['concatenation', 'gated-attention']: 
                scale_outputs.append(x_patches)
    
                if self.deep_supervision: 
                    # (Batch_size, embedding_size) -> (Batch_size, num_classes)
                    deep_spv_out = self.side_classifiers[f'classifier_{scale}'](x_patches)
                    deep_spv_outputs.append(deep_spv_out)
                            
            elif self.type_scale_aggregator in ['max_p', 'mean_p']: 
                deep_spv_out = self.side_classifiers[f'classifier_{scale}'](x_patches)
                deep_spv_outputs.append(deep_spv_out) 

        # --- Scale Aggregation & Final Classification ---
        if self.type_scale_aggregator in ['concatenation', 'gated-attention']:  
            # Stack outputs along a new dimension, num_scales, to create a tensor of shape 
            # (batch_size, num_scales, embedding_size) or (batch_size, num_scales, num_classes)
            x = torch.stack(scale_outputs, dim=1) 
    
            if self.type_scale_aggregator == 'gated-attention':
                # (batch_size, num_scales, embedding_size) --> (batch_size, embedding_size)
                x, A = self.scale_aggregator(x)
                self.save_scale_scores(A)
    
            else: # concatenation scale-aggregator 
                # (batch_size, num_scales, embedding_size) --> (batch_size, num_scales*embedding_size)
                x = self.scale_aggregator(x)
                
            # Apply a MLP to obtain the bag label: (Batch_size, embedding_size) -> (Batch_size, num_classes)
            if not self.is_training and self.type_scale_aggregator == 'concatenation' and x.shape[0] == 1: 
                x, A = self.classifier(x, len(self.scales), self.fcl_encoder_dim, self.is_training)  
                self.save_scale_scores(A)
            else:
                x = self.classifier(x) 

        if self.type_scale_aggregator in ['concatenation', 'gated-attention']: 

            if self.deep_supervision: 
                return x.squeeze(1), deep_spv_outputs # (Batch_size)
 
            return x.squeeze(1)

        return deep_spv_outputs # (Batch_size)
        
class NestedPyramidalMILmodel(MIL):
    def __init__(self, 
                 type_scale_aggregator, 
                 type_region_encoder, 
                 type_region_pooling, 
                 deep_supervision, 
                 scales, 
                 num_inst = None, 
                 mil_args = None) -> None:

        """
        Initialize the Nested Pyramidal Multiple Instance Learning (MIL) model.

        Args:
            type_scale_aggregator (str): method to aggregate multi-scale features 
            type_region_encoder (str): Type of region-level encoder 
            type_region_pooling (str): Pooling method used at the region level 
            deep_supervision (bool): Whether to apply supervision at each scale.
            scales (List[int]): The scales used for multi-scale analysis 
            num_inst (int, optional): Number of instances per bag
            mil_args (dict, optional): Additional MIL arguments passed to the superclass.
        """
        
        self.mil_type = 'embedding'
        self.num_inst = num_inst 
        
        super().__init__(**mil_args)
        
        self.type_scale_aggregator = type_scale_aggregator
        self.deep_supervision = deep_supervision 
        self.scales = scales

        # scale-specific encoders and aggregators
        self.side_inst_aggregator = nn.ModuleDict({
            'encoders': nn.ModuleDict({f'encoder_{scale}': self.MILEncoder(self.embedding_size, self.fcl_encoder_dim, scale_idx) for scale_idx, scale in enumerate(self.scales)}),
            'aggregators': nn.ModuleDict({f'aggregator_{scale}': self.MILAggregator(self.fcl_encoder_dim) for scale in self.scales})
        })

        # shared Region-level encoder and aggregator
        self.region_aggregator = nn.ModuleDict({
            'encoder': self.MILEncoder(self.embedding_size, self.fcl_encoder_dim, -1, type_region_encoder),
            'aggregator': self.MILAggregator(self.fcl_encoder_dim, type_region_pooling)
        })
        
        if self.type_scale_aggregator in ['concatenation', 'gated-attention']: 
            
            if deep_supervision:
                self.side_classifiers = nn.ModuleDict({f'classifier_{scale}': head(self.fcl_encoder_dim, self.num_classes, self.sigmoid_func, self.drop_classhead) for scale in self.scales})
 
            self.scale_aggregator = self.ScaleAggregator(type_scale_aggregator, dim_in = self.fcl_encoder_dim)

            # final classifier head at the multi-scale aggregation level
            if type_scale_aggregator == 'concatenation':
                self.classifier = head(self.fcl_encoder_dim*len(self.scales), self.num_classes, self.sigmoid_func, self.drop_classhead)
            
            else: 
                self.classifier = head(self.fcl_encoder_dim, self.num_classes, self.sigmoid_func, self.drop_classhead) 

                
        elif self.type_scale_aggregator in ['max_p', 'mean_p']:

            self.side_classifiers = nn.ModuleDict({f'classifier_{scale}': head(self.fcl_encoder_dim, self.num_classes, self.sigmoid_func, self.drop_classhead) for scale in self.scales})

        self.inner_scores = {}         
        self.patch_scores = {}
        self.scale_scores = None 

        self.bag_embed = {}
        self.inst_embed = {}

    def ScaleAggregator(self, type_scale_aggregator, dim_in) -> nn.Module:
        """
        Multi-scale aggregator module 
        """

        if type_scale_aggregator in ['concatenation']: 
            scale_aggregator = ConcatAggregator()
        elif type_scale_aggregator == 'attention':
            scale_aggregator =  Attn_Net(dim_in, self.fcl_attention_dim, self.drop_attention_pool, self.map_prob_func)
        elif type_scale_aggregator == 'gated-attention':
            scale_aggregator =  Gated_Attn_Net(dim_in, self.fcl_attention_dim, self.drop_attention_pool, self.map_prob_func)
        else:
            raise ValueError(f"Unknown type_scale_aggregator: {type_scale_aggregator}")
            
        return scale_aggregator

    def save_patch_scores(self, A, s):
        self.patch_scores[s] = A
     
    def save_scale_scores(self, A):
        self.scale_scores = A
        
    def get_patch_scores(self):
        return self.patch_scores

    def get_scale_scores(self):
        return self.scale_scores

    def get_inner_scores(self): 
        return self.inner_scores 

    def save_inner_scores(self, A, scale, region):

        if scale not in self.inner_scores:
            self.inner_scores[scale] = {}
            
        self.inner_scores[scale][region] = A
        
    def forward(self, x, bag_mask=None):
        """
        Forward pass for multi-scale MIL with nested pyramid structure.
        Args:
            x (Tensor or List[Tensor]): Input tensor or list of feature maps for top-down refinement.
            bag_mask (Tensor): Optional attention mask.
        Returns:
            Output logits and optionally deep supervision outputs.
        """

        # --- Instance Feature Extraction ---
        if self.inst_encoder is not None: 

            if isinstance(x, list): # only top-down FPN online refinement over bottom-up backbone feature maps extracted offline 
                    
                batch_size, num_patches, _, _, _ = x[0].size()
                x = [tensor.view(-1, tensor.size(2), tensor.size(3), tensor.size(4)) for tensor in x] 

            else: # online feature extraction 
                    
                # x should be a tensor of size [batch_size, num_patches, C, H, W]
                batch_size, num_patches, C, H, W = x.size()
                
                # Flatten the batch and patches dimensions for feature extraction
                x = x.view(-1, C, H, W)  # New size: [batch_size * num_patches, C, H, W]
                    
            x = self.inst_encoder(x)  # Extract features

            # Reorganize feature maps into pyramid structure
            x_pyramid = OrderedDict() 
            for key, fmap in x.items(): 
                _, channels, height, width = fmap.shape 
                            
                # Transform input: (Batch_size * num_patches, embedding_size, H, W) -> (batch_size, num_patches, embedding_size, H, W)
                fmap = fmap.view(batch_size, num_patches, channels, height, width) 

                x_pyramid[key] = fmap 
                
        if self.type_scale_aggregator in ['concatenation', 'gated-attention']: 
            scale_outputs = []

            if self.deep_supervision: 
                deep_spv_outputs = []
                
        elif self.type_scale_aggregator in ['max_p', 'mean_p']:
            deep_spv_outputs = [] 

        # Process each scale
        for scale in self.scales: 
    
            region_outputs = []

            # Process each patch region
            for region in range(num_patches):
                x_region = x_pyramid[f'feat_{self.scales.index(scale)}'][:, region, :, :] # Get the correct scale feature map

                # nested pixel-level encoding 
                for block_encoder in self.side_inst_aggregator['encoders'][f'encoder_{scale}']:
                    if isinstance(block_encoder, (SetAttentionBlock, InducedSetAttentionBlock)):
                        x_region = block_encoder(x_region, bag_mask)
                    else: 
                        x_region = block_encoder(x_region)

                # nested pixel-level pooling 
                if self.pooling_type in ["attention", "gated-attention", "pma"]:
                    x_region, A = self.side_inst_aggregator['aggregators'][f'aggregator_{scale}'](x_region, bag_mask)
                    self.save_inner_scores(A, scale = scale, region=region)
                                    
                else:
                    x_region = self.side_inst_aggregator['aggregators'][f'aggregator_{scale}'](x_region, bag_mask)
        
                region_outputs.append(x_region)
                    
            # Stack outputs along a new dimension, num_patches, to create a tensor of shape 
            # (batch_size, num_patches, embedding_size) 
            x_patches = torch.stack(region_outputs, dim=1)

            # patch-level encoding 
            for block_encoder in self.region_aggregator['encoder']:
                if isinstance(block_encoder, (SetAttentionBlock, InducedSetAttentionBlock)):
                    x_patches = block_encoder(x_patches, bag_mask)
                else: 
                    x_patches = block_encoder(x_patches)
            
            # patch-level pooling 
            # (Batch_size, N, embedding_size) -> (Batch_size, embedding_size)
            if self.pooling_type in ["attention", "gated-attention", "pma"]:
                x_patches, A = self.region_aggregator['aggregator'](x_patches, bag_mask)
                    
                self.save_patch_scores(A, scale)
            else:
                x_patches = self.region_aggregator['aggregators'](x_patches, bag_mask)

            
            if self.type_scale_aggregator in ['concatenation', 'gated-attention']: 
                scale_outputs.append(x_patches)
    
                if self.deep_supervision: 
                    # (Batch_size, embedding_size) -> (Batch_size, num_classes)
                    deep_spv_out = self.side_classifiers[f'classifier_{scale}'](x_patches)
                    deep_spv_outputs.append(deep_spv_out)
                            
            elif self.type_scale_aggregator in ['max_p', 'mean_p']: 
                deep_spv_out = self.side_classifiers[f'classifier_{scale}'](x_patches)
                deep_spv_outputs.append(deep_spv_out) 

        # multi-scale aggregation 
        if self.type_scale_aggregator in ['concatenation', 'gated-attention']:  
            # Stack outputs along a new dimension, num_scales, to create a tensor of shape 
            # (batch_size, num_scales, embedding_size) or (batch_size, num_scales, num_classes)
            x = torch.stack(scale_outputs, dim=1) 
    
            if self.type_scale_aggregator == 'gated-attention':
                # (batch_size, num_scales, embedding_size) --> (batch_size, embedding_size)
                x, A = self.scale_aggregator(x)
                self.save_scale_scores(A)
    
            else: # concatenation scale-aggregator 
                # (batch_size, num_scales, embedding_size) --> (batch_size, num_scales*embedding_size)
                x = self.scale_aggregator(x)

                
            # Final classification - Apply a MLP to obtain the bag label: (Batch_size, embedding_size) -> (Batch_size, num_classes)
            if not self.is_training and self.type_scale_aggregator == 'concatenation' and x.shape[0] == 1: 
                x, A = self.classifier(x, len(self.scales), self.fcl_encoder_dim, self.is_training)  
                self.save_scale_scores(A)
            else:
                x = self.classifier(x) 

        if self.type_scale_aggregator in ['concatenation', 'gated-attention']: 

            if self.deep_supervision: 
                return x.squeeze(1), deep_spv_outputs # (Batch_size)
 
            return x.squeeze(1)

        return deep_spv_outputs # (Batch_size)

