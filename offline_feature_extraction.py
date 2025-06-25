# internal imports 
from utils.generic_utils import seed_all, save_hdf5
from Datasets.dataset_utils import bags_dataloader 
from FeatureExtractors.mammoclip import load_image_encoder

#external imports 
import warnings
import argparse
import os
import time
from datetime import datetime
from tqdm import tqdm #progress bar

from pathlib import Path

import torch
import pandas as pd
import numpy as np 
import json
import matplotlib.pyplot as plt

def config():
    parser = argparse.ArgumentParser()

    # --- Data Paths ---
    parser.add_argument("--data-dir",default="PreProcessedData/Vindir-mammoclip/VinDir_preprocessed_mammoclip",type=str, help="Path to directory containing data and CSV file")
    parser.add_argument("--img-dir", default="images_png", type=str, help="Directory name for image files inside data-dir")
    parser.add_argument("--csv-file", default="vindr_detection_v1_folds.csv", type=str, help="path to csv file containing metadata")
    parser.add_argument("--clip_chk_pt_path", default=None, type=str, help="Path to Mammo-CLIP chkpt")
    parser.add_argument('--feat_dir', type=str, help='Directory to save extracted features')

    # --- Patch Extraction Settings ---
    parser.add_argument('--patching', action = 'store_true', default = False, help = 'Wether to perform patching on full-resolution images. If false, it will consider previously extracted patches that were saved in a directory (default: False)')
    parser.add_argument('--source_image', type = str, default = 'patches', choices = ['patches', 'full_image'])
    parser.add_argument('--patch_size', type = int, default = 224, help='Patch size for image cropping') 
    #parser.add_argument('--overlap', type = float, default = 0.0)
    parser.add_argument('--overlap', type = float, nargs='*',  default=(0.0), help='Overlap between patches (if any)')
    parser.add_argument('--scales', type=int,  nargs='*',  default=(4, 8, 16, 32), help="List of scales to use for the Feature Pyramid Network (FPN). Default: (2, 4, 16, 32).")
    parser.add_argument('--multi_scale_model', type=str, choices = ['fpn', 'backbone_pyramid', 'image_pyramid'], default = None, help='Type of multiscale model') 
    
    # --- Dataset Settings ---
    parser.add_argument("--img-size", nargs='+', default=[1520, 912], help="Image size in pixels (H, W)")
    parser.add_argument("--data_frac", default=1.0, type=float, help="Fraction of data to be used for training")
    parser.add_argument("--dataset", default="VinDr", type=str, help="Dataset name")
    parser.add_argument("--mean", default=0.3089279, type=float, help="Dataset mean for normalization")
    parser.add_argument("--std", default=0.25053555408335154, type=float, help="Dataset std for normalization")

    # --- Data Augmentation Settings ---
    parser.add_argument("--alpha", default=10, type=float, help="Elastic distortion alpha")
    parser.add_argument("--sigma", default=15, type=float, help="Elastic distortion sigma")
    parser.add_argument("--p", default=1.0, type=float, help="Probability for augmentation")

    # --- Mammo-CLIP settings ---
    parser.add_argument('--model-type', default="Classifier", type=str, help='Model task type')
    parser.add_argument("--arch", default="upmc_breast_clip_det_b5_period_n_ft", type=str,
        help="For b5 classification, [upmc_breast_clip_det_b5_period_n_lp for linear probe and  upmc_breast_clip_det_b5_period_n_ft for finetuning]. "
             "For b2 classification, [upmc_breast_clip_det_b2_period_n_lp for linear probe and  upmc_breast_clip_det_b2_period_n_ft for finetuning].", help="Model architecture name")
    parser.add_argument("--swin_encoder", default="microsoft/swin-tiny-patch4-window7-224", type=str, help="Swin Transformer model identifier"))
    parser.add_argument("--pretrained_swin_encoder", default="y", type=str, help="Whether Swin encoder is pretrained (y/n)")
    parser.add_argument("--swin_model_type", default="y", type=str)
    
    # Device settings 
    parser.add_argument("--num-workers", default=4, type=int, help="Number of data loader workers")
    parser.add_argument("--device", default="cuda", type=str, help="Device to run on")
    parser.add_argument("--apex", default="y", type=str, help="Use Apex mixed-precision training")
    
    # Misc
    parser.add_argument("--seed", default=10, type=int, help="Random seed for reproducibility")
    parser.add_argument("--print-freq", default=5000, type=int, help="How often to print progress")
    parser.add_argument("--log-freq", default=1000, type=int, help="How often to log")

    return parser.parse_args()


def main(args):

    # ------ Setup ------
    torch.cuda.empty_cache() # Clean up
    seed_all(args.seed) # Fix the seed for reproducibility
    
    # Set device and CUDA visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nUsing device:', device)

    # Parse boolean string args
    args.pretrained_swin_encoder = True if args.pretrained_swin_encoder == "y" else False
    args.swin_model_type = True if args.swin_model_type == "y" else False

    # Base model name
    if 'efficientnetv2' in args.arch:
        args.model_base_name = 'efficientv2_s'
    elif 'efficientnet_b5_ns' in args.arch:
        args.model_base_name = 'efficientnetb5'
    else:
        args.model_base_name = args.arch

    args.n_class = 1

    # ------ Load Dataset ------
    args.data_dir = Path(args.data_dir)
    
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)

    # Reduce dataset if desired
    if args.data_frac < 1.0:
        dev_df = args.df[args.df['split'] == "training"].reset_index(drop=True)
        test_df = args.df[args.df['split'] == "test"].reset_index(drop=True)
        
        dev_df = dev_df.sample(frac=args.data_frac, random_state=1, ignore_index=True) 
        
        args.df = pd.concat([dev_df, test_df], ignore_index=True)

    # Initialize data loader
    loader = bags_dataloader(args.df, args)
    print(f"df shape: {args.df.shape}")
    print(args.df.columns)
        
    # ------ Load Mammo-CLIP Model ------
    ckpt = torch.load(args.clip_chk_pt_path, map_location="cpu")
    args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]

    print(ckpt["config"]["model"]["image_encoder"])
    config = ckpt["config"]["model"]["image_encoder"]
    image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"], args.multi_scale_model)

    # Load only image encoder weights
    image_encoder_weights = {}
    for k in ckpt["model"].keys():
        if k.startswith("image_encoder."):
            image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
    image_encoder.load_state_dict(image_encoder_weights, strict=False)
    image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]

    # Freeze encoder weights
    for param in image_encoder.parameters():
        param.requires_grad = False
    
    image_encoder = image_encoder.to(args.device)
    image_encoder.eval()
    
    print(image_encoder_type)
    print(config["name"].lower()) 

    all_num_patches = []

    # ------ Feature Extraction ------
    for count, data in enumerate(tqdm(loader)):
        
        with torch.no_grad():
            inputs = data['x'].to(device, non_blocking=True)

            if args.patching: 
                #inputs = inputs.squeeze(1).permute(0, 1, 4, 2, 3)
                inputs = inputs.squeeze(1)
                batch_size, num_patches, C, H, W = inputs.size()
                all_num_patches.append(num_patches)

                # Flatten the batch and patches dimensions for feature extraction
                inputs = inputs.view(-1, C, H, W)  # New size: [batch_size * num_patches, C, H, W]

                if args.multi_scale_model == 'fpn': 
                    # Extract multi-scale features
                    multi_scale_features = image_encoder(inputs)
                    
                    C4 = multi_scale_features[0]
                    C5 = multi_scale_features[1]

                    C4 = C4.view(batch_size, num_patches, C4.size(1), C4.size(2), C4.size(3)).squeeze(0).cpu() 
                    C5 = C5.view(batch_size, num_patches, C5.size(1), C5.size(2), C5.size(3)).squeeze(0).cpu()
            

                    if count == 0: 
                        print('inputs shape:', inputs.shape) 
                        print('C4 shape:', C4.shape)
                        print('C5 shape:', C5.shape)
                    
                    output_path = os.path.join(args.feat_dir, 'multi_scale', data['patient_id'][0], data['image_id'][0])
                    os.makedirs(output_path, exist_ok=True) 

                    torch.save(C4, os.path.join(output_path, 'C4_patch_features.pt'))
                    torch.save(C5, os.path.join(output_path, 'C5_patch_features.pt'))
                    
                else: 
                    # Extract single-scale patch features
                    features = image_encoder(inputs)

                    # Reshape to [batch_size, num_patches, feature_size]
                    features = features.view(batch_size, num_patches, -1).squeeze(0).cpu() 

                    if count == 0: 
                        print('num_patches:', num_patches) 
                        print('inputs shape:', inputs.shape) 
                        print('features shape:', features.shape)
                        #print("data['coords'] shape:", data['coords'].shape)

                    output_path = os.path.join(args.feat_dir, f'patch_size-{args.scales[0]}', data['patient_id'][0], data['image_id'][0])
                    os.makedirs(output_path, exist_ok=True) 
                    
                    torch.save(features, os.path.join(output_path, 'patch_features.pt'))

                # Save metadata
                asset_dict = {'coords': data['coords']}
                
                attr_dict = {'coords': 
                             {'patch_size': args.patch_size,
                              'scale': args.scales[0], 
                              'overlap': args.overlap,
                              'img_height': args.img_size[0] + data['padding_top'][0] + data['padding_bottom'][0], 
                              'img_width': args.img_size[1] + data['padding_left'][0] + data['padding_right'][0], 
                              'padding_left': data['padding_left'][0], 
                              'padding_top': data['padding_top'][0],
                              'img_dir': output_path
                             }
                            }
                
                save_path_hdf5 = os.path.join(output_path, 'info_patches.h5')
                save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
            
            else: 
                # Extract features from full images
                
                #inputs = inputs.squeeze(1).permute(0, 3, 1, 2)

                image_feature = image_encoder(inputs)
                image_feature = image_feature.flatten(start_dim=1)
                    
                output_path = os.path.join(args.feat_dir, 'Image_features', data['patient_id'][0], data['image_id'][0])
                os.makedirs(output_path, exist_ok=True) 

                torch.save(image_feature, os.path.join(output_path, 'image_feature.pt'))


if __name__ == "__main__":
    args = config()
    main(args)




