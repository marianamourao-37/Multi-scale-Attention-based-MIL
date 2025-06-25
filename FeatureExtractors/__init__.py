# internal imports 
from .mammoclip import load_image_encoder
from .FPN import FeaturePyramidNetwork

# external imports 
import torch 
from torch import Tensor
import torch.nn as nn
from typing import Union, Optional, Callable, Dict

def Define_Feature_Extractor(args) -> Union[nn.Module, int]:
    """
    Loads and configures a feature extractor. 
    For our FPN-MIL models, an FPN-based instance feature extractor is considered. 
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
