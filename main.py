# internal imports 
from MIL.MIL_experiment import do_experiments
from MIL.roi_eval import ROI_Eval
from MIL.inference_MIL_classifier import Eval 
from utils.generic_utils import seed_all

#external imports 
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

warnings.filterwarnings("ignore")
import argparse
import yaml
import time
from datetime import datetime

from pathlib import Path

def config():
    parser = argparse.ArgumentParser()
    
    # Folders
    parser.add_argument('--output_dir', metavar='DIR',default='Mammo-CLIP-output/out_splits_new', help='path to output logs')
    parser.add_argument("--data_dir",default="datasets/Vindir-mammoclip",type=str, help="Path to data file")
    parser.add_argument("--clip_chk_pt_path", default=None, type=str, help="Path to Mammo-CLIP chkpt")
    parser.add_argument("--csv_file", default="grouped_df.csv", type=str, help="data csv file")
    parser.add_argument('--feat_dir', default='new_extracted_features', type=str)
    parser.add_argument("--img_dir", default="VinDir_preprocessed_mammoclip/images_png", type=str, help="Path to image file")

    parser.add_argument('--train', action='store_true', default=False, help='Training mode.')
    parser.add_argument('--evaluation', action='store_true', default=False, help='Evaluation mode.')
    parser.add_argument('--eval_set', default='test', choices = ['val', 'test'], type=str, help="")
    
    # Data settings
    parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    parser.add_argument("--dataset", default="ViNDr", type=str, help="Dataset name.")
    parser.add_argument("--data_frac", default=1.0, type=float, help="Fraction of data to be used for training")
    parser.add_argument("--label", default="Mass", type=str, help="Mass or Suspicious_Calcification")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--n_runs", default=1, type=int)
    parser.add_argument("--start_run", default=0, type=int)
    parser.add_argument('--val_split', type=float, default=0.2, help='val split ratio (default: 0.2)')
    parser.add_argument("--n_folds", default=1, type=int)
    parser.add_argument("--start-fold", default=0, type=int)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)

    # Mammo-CLIP settings 
    parser.add_argument('--model-type', default="Classifier", type=str)
    parser.add_argument("--arch", default="upmc_breast_clip_det_b5_period_n_ft", type=str,
        help="For b5 classification, [upmc_breast_clip_det_b5_period_n_lp for linear probe and  upmc_breast_clip_det_b5_period_n_ft for finetuning]. "
             "For b2 classification, [upmc_breast_clip_det_b2_period_n_lp for linear probe and  upmc_breast_clip_det_b2_period_n_ft for finetuning].")
    parser.add_argument("--swin_encoder", default="microsoft/swin-tiny-patch4-window7-224", type=str)
    parser.add_argument("--pretrained_swin_encoder", default="y", type=str)
    parser.add_argument("--swin_model_type", default="y", type=str)

    parser.add_argument("--feature_extraction", default = 'offline', type = str) 
    parser.add_argument("--feat_dim", default = 352, type = int) 
    
    # Patch extraction 
    parser.add_argument('--patching', action = 'store_true', default = False, help = 'Wether to perform patching on full-resolution images. If false, it will consider previously extracted patches that were saved in a directory (default: False)')
    parser.add_argument('--source_image', type = str, default = 'patches', choices = ['patches', 'full_image'])
    parser.add_argument('--patch_size', type = int, default = 512) 
    parser.add_argument('--overlap', type = float, nargs='*',  default=[0.0])
    
    # MIL model parameters
    parser.add_argument('--mil_type', default=None, choices=[None, 'instance', 'embedding', 'pyramidal_mil'], type=str, help="MIL approach")
    parser.add_argument('--pooling_type', default='mean', choices=['max', 'mean', 'attention', 'gated-attention', 'pma'], type=str, help="MIL pooling operator")
    parser.add_argument('--type_mil_encoder', default='mlp', choices=['mlp', 'sab', 'isab'], type=str, help="Type of MIL encoder.")
    
    parser.add_argument('--fcl_attention_dim', type=int, default=128, metavar='N', help='parameter for attention (internal hidden units)')
    parser.add_argument('--map_prob_func', type=str, default = 'softmax', choices = ['softmax', 'sparsemax', 'entmax', 'alpha_entmax'])

    parser.add_argument('--fcl_encoder_dim', type=int, default=256, help='parameter for set transformer (internal hidden units)')
    parser.add_argument('--sab_num_heads', type=int, default=4, help='parameter for set transformer (number of self-attention heads in set attention blocks)')
    parser.add_argument('--isab_num_heads', type=int, default=4, help='parameter for set transformer (number of self-attention heads in induced set attention blocks)')
    parser.add_argument('--pma_num_heads', type=int, default=1, help='parameter for set transformer (number of self-attention heads in pooling by multihead attention)')
    parser.add_argument('--num_encoder_blocks', type=int, default=2, help='parameter for set transformer (number of encoder layers)')
    parser.add_argument('--trans_num_inds', type=int, default=20, help='parameter for set transformer (number of inducing points for the ISAB)')
    parser.add_argument('--trans_layer_norm', type=bool, default=False)

    #Multi-scale MIL
    parser.add_argument('--multi_scale_model', type=str, choices = ['fpn', 'backbone_pyramid', 'msp'], default = None) 
    parser.add_argument('--scales', type=int,  nargs='*',  default=(16, 32, 64, 128), help="List of scales to use for the multi-scale model.")

    parser.add_argument('--fpn_dim', type=int, default=256)
    parser.add_argument('--upsample_method', type = str, choices = ['bilinear', 'nearest'], default = 'nearest')
    parser.add_argument('--norm_fpn', type = bool, default = False)
    
    parser.add_argument('--deep_supervision', action='store_true', default=False)
    parser.add_argument('--type_scale_aggregator', type=str, choices = ['concatenation', 'max_p', 'mean_p','attention', 'gated-attention'], default=None)

    #Nested MIL 
    parser.add_argument('--nested_model', action='store_true', default=False)
    parser.add_argument('--type_region_aggregator', type=str, choices = ['concatenation', 'max_p', 'mean_p','attention', 'gated-attention'], default=None)
    parser.add_argument('--type_region_encoder', default=None, choices=['mlp', 'sab', 'isab'], type=str, help="Type of MIL encoder.")
    parser.add_argument('--type_region_pooling', default=None, choices=['max', 'mean', 'attention', 'gated-attention', 'pma'], type=str, help="MIL pooling operator")
    
    # Training parameters
    parser.add_argument('--training_mode', type=str, default = 'frozen',
                        help = 'Training mode (default: pretrained)',
                        choices = ['finetune', 'frozen'])
    parser.add_argument('--warmup_stage_epochs', type = int, default = 0)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=9, type=int)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--weighted-BCE", default='n', type=str)
    parser.add_argument('--clip_grad', type=float, default=0.0, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')

    # Data augmentation settings
    parser.add_argument("--balanced-dataloader", default='n', type=str,help='Enable weighted sampling during training (default: False).')
    parser.add_argument("--data_aug", action='store_true', default=False)
    
    parser.add_argument("--alpha", default=10, type=float)
    parser.add_argument("--sigma", default=15, type=float)
    parser.add_argument("--p", default=1.0, type=float)
    
    # LR scheduler settings 
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--warmup-epochs", default=1, type=float)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--num_cycles", default=0.5, type=float)

    # Regularization parameters
    parser.add_argument('--drop_classhead', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the classification head (default: 0.)')
    parser.add_argument('--drop_attention_pool', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the attention pooling mechanism (default: 0.)')
    parser.add_argument('--drop_mha', type=float, default=0.0, metavar='PCT', help='Dropout rate used in the attention pooling mechanism (default: 0.)')
    parser.add_argument('--fcl_dropout', type=float, default=0.0)
    parser.add_argument("--lamda", type=float, default=0.0,
                        help='lambda used for balancing cross-entropy loss and rank loss.')
    
    # ROI evaluation parameters
    parser.add_argument('--roi_eval', action='store_true', default=False, help='Evaluate post-hoc detection performance')
    parser.add_argument('--roi_attention_threshold', type=float, default=0.5)
    parser.add_argument('--visualize_num_images', default=0, type=int, help="")
    parser.add_argument('--quantile_threshold', default = 0.95, type = float) 
    parser.add_argument('--max_bboxes', default = 100, type = int)
    parser.add_argument('--min_area', default = 1024, type = int)
    parser.add_argument('--iou_threshold', default = 0.25, type = float)

    parser.add_argument('--roi_eval_scheme', default='all_roi', choices = ['small_roi', 'medium_roi', 'large_roi', 'all_roi'], type=str, help="")
    parser.add_argument('--roi_eval_set', default='test', choices = ['val', 'test'], type=str, help="")

    parser.add_argument('--iou_method', default = 'iou', choices = ['iou', 'iobb_detection', 'iobb_annotation'], type = str) 
    parser.add_argument('--ap_method', default = 'area', choices = ['area', '11points'], type = str) 
    
    # Device settings 
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)
    
    # Misc
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--print-freq", default=5000, type=int)
    parser.add_argument("--log-freq", default=1000, type=int)
    parser.add_argument("--running-interactive", default='n', type=str)
    parser.add_argument('--eval_scheme', default='kruns_train+val', type=str, help='Evaluation scheme [kruns_train+val | kfold_cv+test ]')
    parser.add_argument('--resume', default = None, type = str) 
    parser.add_argument('--test_example', default = None, type = str) 
    
    return parser.parse_args()


def main(args):
    
    seed_all(args.seed) # Fix the seed for reproducibility
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\ntorch.cuda.current_device():', torch.cuda.current_device())
    print('\nUsing device:', device)

    args.apex = True if args.apex == "y" else False
    
    # get paths
    now = datetime.now().strftime('%Y-%m-%d')
    
    args.running_interactive = True if args.running_interactive == "y" else False

    torch.cuda.empty_cache() # Clean up

    if args.train: 

        # From MammoCLIP's work 
        if args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "mass":
            args.BCE_weights = 15.573306370070778
        elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "suspicious_calcification":
            args.BCE_weights = 37.296728971962615
        
        if args.mil_type: 

            #ENCODER STAGE 
            if args.type_mil_encoder == 'mlp': 
                encoder_text = f'encoder_mlp-dim_{args.fcl_encoder_dim}-dropout_{args.fcl_dropout}'
            else: 
                num_heads = args.sab_num_heads if args.type_mil_encoder == 'sab' else args.isab_num_heads
                layer_norm = 'layer_norm' if args.trans_layer_norm else ''
                encoder_text = f'{args.type_mil_encoder}-dim_{args.fcl_encoder_dim}-nblocks_{args.num_encoder_blocks}-nheads_{num_heads}-dropout_{args.drop_mha}-{layer_norm}-{args.map_prob_func}'

            #POOLING STAGE 
            if args.pooling_type in ['attention', 'gated-attention']:
                pooling_text = f'pooling_{args.pooling_type}-dropout_{args.drop_attention_pool}-{args.map_prob_func}'
            elif args.pooling_type == 'pma': 
                layer_norm = 'layer_norm' if args.trans_layer_norm else ''
                pooling_text = f'pooling_{args.pooling_type}-dropout_{args.drop_mha}-{layer_norm}-{args.map_prob_func}'
            else: 
                pooling_text = f'pooling_{args.pooling_type}'

            #MULTI-SCALE INSTANCE ENCODER 
            if args.multi_scale_model in ['fpn', 'backbone_pyramid']: 
                deep_supervision = f'-deep_supervision' if args.deep_supervision else ''
                nested_model = f'-nested' if args.nested_model else ''
                multi_scale_text = f'{args.multi_scale_model}{nested_model}{deep_supervision}'

                scale_aggregator_text = f'scale_aggregator_{args.type_scale_aggregator}' 
            
                root = f"multi_scale/{multi_scale_text}/{scale_aggregator_text}/{encoder_text}/{pooling_text}"
                
            elif args.multi_scale_model == 'msp':
                deep_supervision = '-deep_supervision' if args.deep_supervision else '' 
                root = f"multi_scale/MSP{deep_supervision}/scale_aggregator_{args.type_scale_aggregator}/{encoder_text}/{pooling_text}"
            
            else:  
                root = f"single_scale-patch_size_{args.scales[0]}/{args.mil_type}/{encoder_text}/{pooling_text}"

        if args.feature_extraction == 'online': 
            data_aug = '-aug' if args.data_aug else '' 
            train_mode = f'ft-warmup_{args.warmup_stage_epochs}{data_aug}' if args.training_mode == 'finetune' else f'patch_size_{args.patch_size}-overlap_{args.overlap[0]}_lp{data_aug}'
        else: 
            train_mode = 'offline_feature_extraction'
            
        args.output_path = Path(f"{args.output_dir}/MIL_experiments/{args.dataset}_data_frac_{args.data_frac}/{args.label}/{train_mode}/{root}/{now}") 
        
        os.makedirs(args.output_path, exist_ok=True)
        print(f"output_path: {args.output_path}")
    
        # Convert any PosixPath in args.__dict__ to a string
        args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in args.__dict__.items()}

        # Cache the args as a text string to save them in the output dir 
        args_text = yaml.safe_dump(args_dict, default_flow_style=False)

        with open(os.path.join(args.output_path, "args.yaml"), 'w') as f:
            f.write(args_text)
        
        do_experiments(args, device)
        
    elif args.evaluation: 
        Eval(args, device) 

    elif args.roi_eval: 
        ROI_Eval(args, device) 

if __name__ == "__main__":
    args = config()
    main(args)
