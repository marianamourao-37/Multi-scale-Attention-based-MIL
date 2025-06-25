# external imports 
import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Optional, Callable, Dict
from collections import OrderedDict

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) module for multi-scale instance feature extraction.
    This implementation is based on "Feature Pyramid Network for Object Detection" (https://arxiv.org/abs/1612.03144).
    Adapted from https://github.com/pytorch/vision/blob/release/0.12/torchvision/ops/feature_pyramid_network.py

    Args:
        backbone (nn.Module): Backbone network that provides feature maps.
        scales (list[int]): List of scales to be used for the FPN.
        out_channels (int): Number of channels for the FPN output.
        top_down_pathway (bool): Whether to use top-down pathway.
        upsample_method (str): Interpolation method for upsampling ("nearest", "bilinear", etc.).
        norm_layer (callable, optional): Normalization layer to use. Default: None.
    """
    
    def __init__(
        self,
        backbone, 
        scales, 
        out_channels: int,
        top_down_pathway: bool = True,
        upsample_method: str = 'nearest', 
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super().__init__()
                
        self.top_down_pathway = top_down_pathway
        self.upsample_method = upsample_method

        self.backbone = backbone
        in_channels_list = [120, 352]
        
        if norm_layer: 
            norm_layer = nn.GroupNorm(num_groups = 1, num_channels = out_channels)
            use_bias = False
        else:
            norm_layer = nn.Identity()
            use_bias = True

        
        self.inner_blocks = nn.ModuleDict({f"inner_block_{idx}": 
                                           nn.Sequential(
                                               nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias), 
                                               norm_layer
                                           ) for idx, in_channels in enumerate(in_channels_list)
                                          })
            

        self.layer_blocks = nn.ModuleDict({f"layer_block_{idx}": 
                                           nn.Sequential(
                                               nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias), 
                                               norm_layer
                                           ) for idx in range(len(in_channels_list))
                                          })

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Computes the FPN output for the input image.

        Args:
            x (Tensor): Input image tensor or precomputed feature maps.

        Returns:
            results (OrderedDict[Tensor]): Feature maps at different pyramid levels
        """
        
        if self.backbone is not None: 
            # Extract online feature maps from the backbone 
            selected_fmaps = self.backbone(x)
        else: # offline pre-extracted feature maps 
            selected_fmaps = x 
        
        # Initialize the last inner feature map from the top feature map
        last_inner = self.inner_blocks[f"inner_block_{len(selected_fmaps) - 1}"](selected_fmaps[-1])

        # Create results list and initialize it with the last inner feature map
        results = [self.layer_blocks[f"layer_block_{len(selected_fmaps) - 1}"](last_inner)]
        
        # Build the top-down pathway if enabled
        if self.top_down_pathway:
            for idx in range(len(selected_fmaps) - 2, -1, -1):
                # Process inner lateral connections
                inner_lateral = self.inner_blocks[f"inner_block_{idx}"](selected_fmaps[idx])

                # Compute the spatial size of the feature map
                feat_shape = inner_lateral.shape[-2:]

                # Upsample the last inner feature map 
                inner_top_down = F.interpolate(last_inner, 
                                               size=feat_shape, 
                                               mode=self.upsample_method, 
                                              )
                
                # Merge and update current level
                last_inner = inner_lateral + inner_top_down

                # # Apply 3x3 conv on merged feature map
                results.insert(0, self.layer_blocks[f"layer_block_{idx}"](last_inner))

        else:
            for idx in range(len(selected_fmaps) - 2, -1, -1):
                # Process inner lateral connections without top-down pathway
                inner_lateral = self.inner_blocks[f"inner_block_{idx}"](selected_fmaps[idx])

                # Insert the result at the beginning of the results list
                results.insert(0, self.layer_blocks[f"layer_block_{idx}"](inner_lateral))

        # stride four downsampling over the coarser feature maps 
        results.append(F.max_pool2d(results[-1], kernel_size=1, stride=4, padding=0))
                       
        # Convert the results list to an OrderedDict 
        results = OrderedDict([(f'feat_{idx}', fmap) for idx, fmap in enumerate(results)])
                
        return results
