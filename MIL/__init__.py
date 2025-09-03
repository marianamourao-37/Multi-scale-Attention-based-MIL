import torch 
import math

# internal imports 
from FeatureExtractors import Define_Feature_Extractor, FeaturePyramidNetwork
from .MILmodels import EmbeddingMIL, PyramidalMILmodel, NestedPyramidalMILmodel

def build_model(args): 

    ############################ Define the Feature Extractor ############################
    if args.feature_extraction == 'online': 
        feature_extractor, num_chs = Define_Feature_Extractor(args) 

    else: # offline feature extraction: use pre-extracted features 
            
        if args.multi_scale_model in ['backbone_pyramid', 'fpn']: # FPN-based instance encoder for fpn-based mil models
            
            feature_extractor = FeaturePyramidNetwork(
                backbone=None, 
                scales=args.scales,                            
                out_channels=args.fpn_dim,                           
                top_down_pathway = True if args.multi_scale_model == 'fpn' else False,                                    
                upsample_method = args.upsample_method,      
                norm_layer = args.norm_fpn
            )
                
            num_chs = args.fpn_dim 
            
        else: # if single/multi-scale patch-based mil models 
            feature_extractor = None # directly use the pre-extracted features 
            num_chs = args.feat_dim # feat dim of pre-extracted features 
           
    ############################ Define the MIL Model ############################            
    mil_args = dict(is_training = args.train, #not args.roi_eval, 
                    multi_scale_model = args.multi_scale_model,  
                    inst_encoder=feature_extractor,
                    embedding_size=num_chs,
                    sigmoid_func = False, 
                    num_classes=args.n_class,
                    drop_classhead=args.drop_classhead,
                    map_prob_func = args.map_prob_func,
                    # MIL Encoder args
                    type_mil_encoder = args.type_mil_encoder,
                    fcl_encoder_dim = args.fcl_encoder_dim, 
                    fcl_dropout = args.fcl_dropout if args.type_mil_encoder == 'mlp' else None, 
                    sab_num_heads = args.sab_num_heads if args.type_mil_encoder == 'sab' else None, 
                    isab_num_heads = args.isab_num_heads if args.type_mil_encoder == 'isab' else None,
                    num_encoder_blocks = args.num_encoder_blocks if args.type_mil_encoder in ['sab', 'isab'] else None, 
                    # MIL Aggregator args
                    pooling_type=args.pooling_type,
                    fcl_attention_dim=args.fcl_attention_dim,
                    drop_attention_pool=args.drop_attention_pool,
                    pma_num_heads = args.pma_num_heads if args.pooling_type == 'pma' else None, 
                    # General self-attention based args 
                    drop_mha=args.drop_mha if args.type_mil_encoder in ['isab', 'sab'] else None, 
                    trans_layer_norm=args.trans_layer_norm if args.type_mil_encoder in ['isab', 'sab'] else None
                   )

    # instantiate MIL Model
    if args.mil_type == 'embedding': # single-scale patch-based mil models 
        model = EmbeddingMIL(mil_type = args.mil_type, 
                             num_inst = [math.ceil(args.img_size[0]/s) * math.ceil(args.img_size[1]/s) for s in args.scales], # number of instances (patches) per image 
                             mil_args = mil_args)
        
    elif args.mil_type == 'pyramidal_mil':

        if args.nested_model: # nested MIL formulation (multi-level aggregation) 
            
            # number of instances per scale for each aggregation level 
            num_inst = [(args.patch_size/s)**2 for s in args.scales]
            num_inst.append(math.ceil(args.img_size[0]/args.patch_size) * math.ceil(args.img_size[1]/args.patch_size))
            
            model = NestedPyramidalMILmodel(
                args.type_scale_aggregator, 
                args.type_region_encoder, 
                args.type_region_pooling, 
                deep_supervision = args.deep_supervision, 
                scales = args.scales, 
                num_inst = num_inst,
                mil_args = mil_args
            )
            
        else: # convetional MIL formulation (globally group and aggregate all instances under the same bag for each scale) 

            # number of instances per scale
            if args.multi_scale_model in ['fpn', 'backbone_pyramid']: # FPN-based mil models 
                num_patches = math.ceil(args.img_size[0]/args.patch_size) * math.ceil(args.img_size[1]/args.patch_size)
                num_inst = [(args.patch_size/s)**2 * num_patches for s in args.scales] 
                
            else: # multi-scale patch-based mil models 
                num_inst = [math.ceil(args.img_size[0]/s) * math.ceil(args.img_size[1]/s) for s in args.scales]
            
            model = PyramidalMILmodel(
                args.type_scale_aggregator, 
                args.deep_supervision, 
                args.scales, 
                num_inst = num_inst,
                mil_args = mil_args
            )

    return model 
