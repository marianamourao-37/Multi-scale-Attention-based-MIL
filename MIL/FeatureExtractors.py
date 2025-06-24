# internal imports 
from mammoclip import load_image_encoder

# external imports 
from collections import OrderedDict

import torch 
from torch import Tensor
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
import torchvision.models as models

from typing import Union, Optional, Callable, Dict

import math 
        
class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) module for multi-scale instance feature extraction.
    This implementation is based on "Feature Pyramid Network for Object Detection" (https://arxiv.org/abs/1612.03144).

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

def Define_Feature_Extractor(args) -> Union[nn.Module, int]:
    """
    Loads and configures a feature extractor with optional FPN wrapping for multi-scale instance feature extraction.

    """

    # Load CLIP checkpoint containing pretrained image encoder and config
    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu")
    
    # Extract image encoder type and configuration from checkpoint
    args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]
    print(ckpt["config"]["model"]["image_encoder"])
    config = ckpt["config"]["model"]["image_encoder"]

    # Instantiate the image encoder model
    feature_extractor = load_image_encoder(ckpt["config"]["model"]["image_encoder"], args.multi_scale_model)

    # load pretrained weights into the encoder
    image_encoder_weights = {}
    for k in ckpt["model"].keys():
        if k.startswith("image_encoder."):
            image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
    feature_extractor.load_state_dict(image_encoder_weights, strict=False)
    image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
            
    num_chs = args.feat_dim 

    # If FPN-based instance encoder 
    if args.multi_scale_model in ['fpn', 'backbone_pyramid']: 
          
        feature_extractor = FeaturePyramidNetwork(
            backbone=feature_extractor, 
            scales=args.scales,                            
            out_channels=args.fpn_dim,                           
            top_down_pathway = True if args.multi_scale_model == 'fpn' else False,                                    
            upsample_method = args.upsample_method,      
            norm_layer = args.norm_fpn
        )
        
        num_chs = args.fpn_dim
    
    return feature_extractor, num_chs
