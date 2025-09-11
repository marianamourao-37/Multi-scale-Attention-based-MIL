#external imports 
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# internal imports 
from Datasets.dataset_utils import MIL_dataloader
from MIL import build_model 

from utils.metrics import auroc, evaluate_metrics
from utils.generic_utils import seed_all, AverageMeter, timeSince, print_network, clear_memory 
from utils.training_setup_utils import initialize_training_setup, Training_Stage_Config
from utils.plot_utils import plot_loss_and_acc_curves, plot_lrs_scheduler, plot_confusion_matrix, ROC_curves
from utils.data_split_utils import generator_cross_val_folds, stratified_train_val_split

def do_experiments(args, device):
        
    args.n_class = 1 # Binary classification setup (single output neuron)
        
    # Define class labels based on selected task
    if args.label.lower() == 'mass':
        class0 = 'not_mass'
        class1 = 'mass'
    elif args.label.lower() == 'suspicious_calcification':
        class0 = 'not_calcification'
        class1 = 'calcification'   

    label_dict = {class0: 0, class1: 1}

    ############################ Data Setup ############################
    args.data_dir = Path(args.data_dir)
    
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    
    print(f"df shape: {args.df.shape}")
    print(args.df.columns)

    # Split data into dev (train+val) and test sets
    dev_df = args.df[args.df['split'] == "training"].reset_index(drop=True)
    test_df = args.df[args.df['split'] == "test"].reset_index(drop=True)

    # reduce dataset size for debugging/experiments if desired
    if args.data_frac < 1.0:
        dev_df = dev_df.sample(frac=args.data_frac, random_state=1, ignore_index=True) 

    # repeated k runs using fixed data splits 
    if args.eval_scheme == 'kruns_train+val+test': 

        # split development set into training and validation sets
        train_df, val_df = stratified_train_val_split(dev_df, 0.2, args = args)

        # initialize results dictionary based on model type
        if args.multi_scale_model is not None: 

            # track results for each scale if required by model configuration
            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:  
                all_val_results = {scale: {'f1': [], 'bacc': [], 'auc_roc': []} for scale in args.scales}
                all_test_results = {scale: {'f1': [], 'bacc': [], 'auc_roc': []} for scale in args.scales}

            else: 
                all_val_results = {}
                all_test_results = {}

            # track aggregated results
            all_val_results['aggregated'] = {'f1': [], 'bacc': [], 'auc_roc': []}
            all_test_results['aggregated'] = {'f1': [], 'bacc': [], 'auc_roc': []}
        
        else: 
            # track results for non multi-scale models 
            all_val_results = {'f1': [], 'bacc': [], 'auc_roc': []}
            all_test_results = {'f1': [], 'bacc': [], 'auc_roc': []} 
            
        # set test data loader
        test_loader = MIL_dataloader(test_df ,'test', args)

        # perform multiple runs (kruns) of training and testing
        for idx_run in range(args.n_runs):
            print(f'\n================== run nº: {idx_run} ======================')
            args.cur_fold = idx_run  

            # set seed for reproducibility
            seed_all(args.seed+args.start_run+idx_run)

            # create directory for saving results for this run
            path_results_run = args.output_path / f'run_{args.start_run+idx_run}'
            Path(path_results_run).mkdir(parents=True, exist_ok=True)

            # train and validate model
            val_results, best_checkpoint_path = k_experiment(train_df, val_df, output_path= path_results_run, args = args, device = device)

            # load the best model checkpoint
            checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
            fold_model = build_model(args)
            fold_model.load_state_dict(checkpoint['model'])
            fold_model.to(device)

            # evaluate model on test set
            test_targs, test_preds, test_probs, test_results = valid_fn(
                test_loader, fold_model, criterion = torch.nn.BCEWithLogitsLoss(reduction='mean'), args = args, device = device, split = 'test')

            # free GPU memory
            del fold_model; clear_memory()

            # report and store test results
            if args.multi_scale_model is not None: 
                print(f"\nTest Loss: {test_results['loss']:.4f}") 

                # Print results for individual scales if applicable
                if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:  
                    for s in args.scales:
                        print(f"Scale: {s} --> Test F1-Score: {test_results[s]['f1']:.4f} | Test Bacc: {test_results[s]['bacc']:.4f} | Test ROC-AUC: {test_results[s]['auc_roc']:.4f}")            

                # Print aggregated results
                print(f"Aggregated Results --> Test F1-Score: {test_results['aggregated']['f1']:.4f} | Test Bacc: {test_results['aggregated']['bacc']:.4f} | Test ROC-AUC: {test_results['aggregated']['auc_roc']:.4f}")

                # Generate confusion matrix and ROC curves
                plot_confusion_matrix(test_results['aggregated']['cf_matrix'], label_dict, '', path_results_run)
                ROC_curves(test_targs, test_probs, '', path_results_run)

                # Append results per scale
                if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:  
                
                    for s in args.scales:
                        all_val_results[s]['f1'].append(val_results[s]['f1'])
                        all_val_results[s]['bacc'].append(val_results[s]['bacc'])
                        all_val_results[s]['auc_roc'].append(val_results[s]['auc_roc'])
        
                        all_test_results[s]['f1'].append(test_results[s]['f1'])
                        all_test_results[s]['bacc'].append(test_results[s]['bacc'])
                        all_test_results[s]['auc_roc'].append(test_results[s]['auc_roc'])

                # Append aggregated results
                all_test_results['aggregated']['f1'].append(test_results['aggregated']['f1'])
                all_test_results['aggregated']['bacc'].append(test_results['aggregated']['bacc'])
                all_test_results['aggregated']['auc_roc'].append(test_results['aggregated']['auc_roc'])
                
                all_val_results['aggregated']['f1'].append(val_results['aggregated']['f1'])
                all_val_results['aggregated']['bacc'].append(val_results['aggregated']['bacc'])
                all_val_results['aggregated']['auc_roc'].append(val_results['aggregated']['auc_roc'])

            else: 
                # Log and store results for non-multiscale models
                
                print(f"Test F1-Score: {test_results['f1']:.4f} | Test Bacc: {test_results['bacc']:.4f} | Test ROC-AUC: {test_results['auc_roc']:.4f}")           

                plot_confusion_matrix(test_results['cf_matrix'], label_dict, '', path_results_run)
                ROC_curves(test_targs, test_probs, '', path_results_run)
                
                # Append Results 
                all_val_results['f1'].append(val_results['f1'])
                all_val_results['bacc'].append(val_results['bacc'])
                all_val_results['auc_roc'].append(val_results['auc_roc'])
    
                all_test_results['f1'].append(test_results['f1'])
                all_test_results['bacc'].append(test_results['bacc'])
                all_test_results['auc_roc'].append(test_results['auc_roc'])

        # Collect all results into structured format
        val_results_data = {'runs': np.arange(args.n_runs)}
        test_results_data = {'runs': np.arange(args.n_runs)}

        if args.multi_scale_model is not None: 

            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
                
                # Append metrics for all scales
                for s in args.scales:
                    val_results_data[f'bacc_{s}'] = all_val_results[s]['bacc']
                    val_results_data[f'f1_{s}'] = all_val_results[s]['f1']
                    val_results_data[f'auc_roc_{s}'] = all_val_results[s]['auc_roc']
        
                    test_results_data[f'bacc_{s}'] = all_test_results[s]['bacc']
                    test_results_data[f'f1_{s}'] = all_test_results[s]['f1']
                    test_results_data[f'auc_roc_{s}'] = all_test_results[s]['auc_roc']
            
            # Append metrics for aggregated results
            val_results_data['bacc_aggregated'] = all_val_results['aggregated']['bacc']
            val_results_data['f1_aggregated'] = all_val_results['aggregated']['f1']
            val_results_data['auc_roc_aggregated'] = all_val_results['aggregated']['auc_roc']
    
            test_results_data['bacc_aggregated'] = all_test_results['aggregated']['bacc']
            test_results_data['f1_aggregated'] = all_test_results['aggregated']['f1']
            test_results_data['auc_roc_aggregated'] = all_test_results['aggregated']['auc_roc']
            
        else: 
            val_results_data['bacc'] = all_val_results['bacc']
            val_results_data['f1'] = all_val_results['f1']
            val_results_data['auc'] = all_val_results['auc_roc']
    
            test_results_data['bacc'] = all_test_results['bacc']
            test_results_data['f1'] = all_test_results['f1']
            test_results_data['auc'] = all_test_results['auc_roc']
            
        # Create the final DataFrame
        val_results_data = pd.DataFrame(val_results_data)
        test_results_data = pd.DataFrame(test_results_data)
        
        if args.n_runs > 1: 
            
            # Calculate mean and std for specific columns
            val_mean_std = val_results_data.drop('runs', axis=1).agg(['mean', 'std']).reset_index(drop=True)
            test_mean_std = test_results_data.drop('runs', axis=1).agg(['mean', 'std']).reset_index(drop=True)
            val_mean_std['runs'] = ['mean', 'std']
            test_mean_std['runs'] = ['mean', 'std']

            # Append mean and std to the original DataFrame
            val_results_data = pd.concat([val_results_data, val_mean_std]).reset_index(drop=True)
            test_results_data = pd.concat([test_results_data, test_mean_std]).reset_index(drop=True)

        # Combine validation and test results
        metrics_data = pd.concat([val_results_data, test_results_data], keys=['validation', 'test'], names=['split', 'index'])
        metrics_data = metrics_data.reset_index(level='split') # Reset index to turn the keys into columns
        metrics_data.to_csv(args.output_path / 'results_summary.csv', index=False)


    elif args.eval_scheme == 'kfold_cv+test':

        # Generate k-fold cross-validation splits 
        train_val_splits = generator_cross_val_folds(dev_df, args.n_folds, args.label) 

        # Initialize result dictionaries 
        all_val_results = {scale: {'f1': [], 'bacc': [], 'auc_roc': []} for scale in args.scales}
        all_val_results['aggregated'] = {'f1': [], 'bacc': [], 'auc_roc': []}

        all_test_results = {scale: {'f1': [], 'bacc': [], 'auc_roc': []} for scale in args.scales}
        all_test_results['aggregated'] = {'f1': [], 'bacc': [], 'auc_roc': []}

        # initialize test dataloader 
        test_loader = MIL_dataloader(test_df ,'test', args)
        
        checkpoint_folds = []

        # Iterate through each fold
        for fold in range(args.start_fold, args.n_folds):

            print(f'\n================== fold: {fold} training ======================')
            
            args.cur_fold = fold
            seed_all(args.seed)

            # Setup path for the current fold's results 
            path_results_fold = args.output_path / f'fold_{fold}'
            Path(path_results_fold).mkdir(parents=True, exist_ok=True)

            # Get the next train/val split
            train_df, val_df = next(train_val_splits)

            # Train and evaluate on val set
            val_results, best_checkpoint_path = k_experiment(train_df, val_df, path_results_fold, args, device)

            # Log validation results
            print(f"\nVal Loss: {val_results['loss']:.4f}")        
            for s in args.scales:
                print(f"Scale: {s} --> Val F1-Score: {val_results[s]['f1']:.4f} | Val Bacc: {val_results[s]['bacc']:.4f} | Val ROC-AUC: {val_results[s]['auc_roc']:.4f}")            
            
            print(f"Aggregated Results --> Val F1-Score: {val_results['aggregated']['f1']:.4f} | Val Bacc: {val_results['aggregated']['bacc']:.4f} | Val ROC-AUC: {val_results['aggregated']['auc_roc']:.4f}")

            # ****** Store val metrics ******
            for s in args.scales:
                all_val_results[s]['f1'].append(val_results[s]['f1'])
                all_val_results[s]['bacc'].append(val_results[s]['bacc'])
                all_val_results[s]['auc_roc'].append(val_results[s]['auc_roc'])
            
            all_val_results['aggregated']['f1'].append(val_results['aggregated']['f1'])
            all_val_results['aggregated']['bacc'].append(val_results['aggregated']['bacc'])
            all_val_results['aggregated']['auc_roc'].append(val_results['aggregated']['auc_roc'])

            # Load best checkpoint model
            checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
            fold_path = checkpoint['dir_path']

            fold_model = build_model(args)
            fold_model.load_state_dict(checkpoint['model'])
            fold_model.to(device)

            # Evaluate on test set
            test_targs, test_preds, test_probs, test_results = valid_fn(
                test_loader, fold_model, criterion = torch.nn.BCEWithLogitsLoss(reduction='mean'), args = args, device = device, split = 'test')

            del fold_model; clear_memory()

            # Log test results
            print(f"\nTest Loss: {test_results['loss']:.4f}")        
            for s in args.scales:
                print(f"Scale: {s} --> Test F1-Score: {test_results[s]['f1']:.4f} | Test Bacc: {test_results[s]['bacc']:.4f} | Test ROC-AUC: {test_results[s]['auc_roc']:.4f}")            
            
            print(f"Aggregated Results --> Test F1-Score: {test_results['aggregated']['f1']:.4f} | Test Bacc: {test_results['aggregated']['bacc']:.4f} | Test ROC-AUC: {test_results['aggregated']['auc_roc']:.4f}")

            # Store test metrics
            for s in args.scales:
                all_test_results[s]['f1'].append(test_results[s]['f1'])
                all_test_results[s]['bacc'].append(test_results[s]['bacc'])
                all_test_results[s]['auc_roc'].append(test_results[s]['auc_roc'])

            all_test_results['aggregated']['f1'].append(test_results['aggregated']['f1'])
            all_test_results['aggregated']['bacc'].append(test_results['aggregated']['bacc'])
            all_test_results['aggregated']['auc_roc'].append(test_results['aggregated']['auc_roc'])

            # Save confusion matrix and ROC curves
            plot_confusion_matrix(test_results['aggregated']['cf_matrix'], label_dict, '', fold_path)
            ROC_curves(test_targs, test_probs, '', fold_path)

        # Create a dictionary to hold all final results
        val_results_data = {'folds': np.arange(args.n_folds)}
        test_results_data = {'folds': np.arange(args.n_folds)}
        
        # Append metrics for all scales
        for s in args.scales:
            val_results_data[f'bacc_{s}'] = all_val_results[s]['bacc']
            val_results_data[f'f1_{s}'] = all_val_results[s]['f1']
            val_results_data[f'auc_roc_{s}'] = all_val_results[s]['auc_roc']

            test_results_data[f'bacc_{s}'] = all_test_results[s]['bacc']
            test_results_data[f'f1_{s}'] = all_test_results[s]['f1']
            test_results_data[f'auc_roc_{s}'] = all_test_results[s]['auc_roc']
        
        # Append metrics for aggregated results
        val_results_data['bacc_aggregated'] = all_val_results['aggregated']['bacc']
        val_results_data['f1_aggregated'] = all_val_results['aggregated']['f1']
        val_results_data['auc_roc_aggregated'] = all_val_results['aggregated']['auc_roc']

        test_results_data['bacc_aggregated'] = all_test_results['aggregated']['bacc']
        test_results_data['f1_aggregated'] = all_test_results['aggregated']['f1']
        test_results_data['auc_roc_aggregated'] = all_test_results['aggregated']['auc_roc']
        
        # Create the final DataFrame
        val_results_data = pd.DataFrame(val_results_data)
        test_results_data = pd.DataFrame(test_results_data)

        # Compute mean and std if multiple folds
        if args.n_folds > 1: 
            
            # Calculate mean and std for specific columns
            val_mean_std = val_results_data.drop('folds', axis=1).agg(['mean', 'std']).reset_index(drop=True)
            test_mean_std = test_results_data.drop('folds', axis=1).agg(['mean', 'std']).reset_index(drop=True)
            val_mean_std['folds'] = ['mean', 'std']
            test_mean_std['folds'] = ['mean', 'std']

            # Append mean and std to the original DataFrame
            val_results_data = pd.concat([val_results_data, val_mean_std]).reset_index(drop=True)
            test_results_data = pd.concat([test_results_data, test_mean_std]).reset_index(drop=True)

        # Save results to CSV
        metrics_data = pd.concat([val_results_data, test_results_data], keys=['validation', 'test'], names=['split', 'index'])
        metrics_data = metrics_data.reset_index(level='split') # Reset index to turn the keys into columns
        metrics_data.to_csv(args.output_path / 'results_summary.csv', index=False)


def k_experiment(train_df, val_df, output_path, args, device): 
    """
    Executes a single train/validation experiment.
    
    Args:
        train_df (DataFrame): Training data.
        val_df (DataFrame): Validation data.
        output_path (Path): Directory to save results and checkpoints.
        args (Namespace): Configuration and hyperparameters.
        device (torch.device): Device to run model on.

    Returns:
        Tuple:
            - best_val_stats (dict): Best evaluation metrics on validation set.
            - best_model_path (str): Path to the best model checkpoint.
    """
        
    if args.running_interactive:
        # test on small subsets of data on interactive mode
        train_df = train_df.sample(1000)
        val_df = val_df.sample(n=1000)

    # Initialize data loaders
    train_loader = MIL_dataloader(train_df, 'train', args)
    valid_loader = MIL_dataloader(val_df ,'val', args)
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}')

    # Build and load model
    model = build_model(args)
    print("Model is loaded")

    # Setup training stage manager if online feature extraction is enabled
    training_stage_manager = Training_Stage_Config(model=model, training_mode=args.training_mode, warmup_epochs=args.warmup_stage_epochs) if args.feature_extraction == 'online' else None 

    model = model.to(device)
    print_network(model)

    optimizer, scheduler, scaler, train_criterion, eval_criterion = initialize_training_setup(train_loader, model, device, args)

    best_val_stats, best_model = train_loop(train_loader, valid_loader, model, training_stage_manager, train_criterion, eval_criterion, optimizer, scheduler, scaler, output_path, args, device)
    
    return best_val_stats, best_model
    

def train_loop(train_loader, valid_loader, model, training_stage_manager, train_criterion, eval_criterion, optimizer, scheduler, scaler, output_path, args, device):

    best_aucroc = 0.
    best_epoch = 0 

    # Dictionaries to keep track of training and validation metrics per epoch
    train_results = {'loss': [], 'f1': [], 'bacc': [], 'auc_roc':[], 'lr':[]}
    val_results = {'loss': [], 'f1': [], 'bacc': [], 'auc_roc':[]}
        
    for epoch in range(args.epochs):

        print(f"\n-------- Epoch {epoch + 1}/{args.epochs} --------")
        
        start_time = time.time()

        if training_stage_manager is not None:
            training_stage_manager(model, optimizer, epoch, optimizer.param_groups[0]['lr'])

        # training for one epoch
        train_stats = train_fn(train_loader, model, train_criterion, optimizer, epoch, args, scheduler, scaler, device)

        # validation after the epoch
        val_stats = valid_fn(valid_loader, model, eval_criterion, args, device, split = 'val', epoch = epoch)
    
        elapsed = time.time() - start_time

        # If using multi-scale model, report scale-specific and aggregated results
        if args.multi_scale_model is not None: 
            print(f"\nTrain Loss: {train_stats['loss']:.4f}")

            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
                for s in args.scales:
                    print(f"Scale: {s} --> Train F1-Score: {train_stats[s]['f1']:.4f} | Train Bacc: {train_stats[s]['bacc']:.4f} | Train ROC-AUC: {train_stats[s]['auc_roc']:.4f}")
                
            print(f"Aggregated Results --> Train F1-Score: {train_stats['aggregated']['f1']:.4f} | Train Bacc: {train_stats['aggregated']['bacc']:.4f} | Train ROC-AUC: {train_stats['aggregated']['auc_roc']:.4f}")
        
            print(f"\nVal Loss: {val_stats['loss']:.4f}") 

            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
                for s in args.scales:
                    print(f"Scale: {s} --> Val F1-Score: {val_stats[s]['f1']:.4f} | Val Bacc: {val_stats[s]['bacc']:.4f} | Val ROC-AUC: {val_stats[s]['auc_roc']:.4f}")            
            
            print(f"Aggregated Results --> Val F1-Score: {val_stats['aggregated']['f1']:.4f} | Val Bacc: {val_stats['aggregated']['bacc']:.4f} | Val ROC-AUC: {val_stats['aggregated']['auc_roc']:.4f}")
        
            # Update results dictionary
            train_results['loss'].append(train_stats['loss'])
            train_results['f1'].append(train_stats['aggregated']['f1'])
            train_results['bacc'].append(train_stats['aggregated']['bacc'])
            train_results['auc_roc'].append(train_stats['aggregated']['auc_roc'])
            train_results['lr'].append(train_stats['lr'])
                
            val_results['loss'].append(val_stats['loss'])
            val_results['f1'].append(val_stats['aggregated']['f1'])
            val_results['bacc'].append(val_stats['aggregated']['bacc'])
            val_results['auc_roc'].append(val_stats['aggregated']['auc_roc'])

            # Save checkpoint if best validation ROC-AUC so far
            if best_aucroc < val_stats['aggregated']['auc_roc']:
                best_aucroc = val_stats['aggregated']['auc_roc']
                best_val_stats = val_stats 
    
                best_epoch = epoch + 1
                              
                model_name = 'best_model.pth'
                best_checkpoint_path = output_path / model_name
                
                print(f'\nEpoch {epoch + 1} - Save aucroc: {best_aucroc:.4f} Model')
                    
                torch.save(
                    { 
                        'model': model.state_dict(),
                        #'predictions': val_predictions,
                        'epoch': epoch,
                        'auroc': val_stats['aggregated']['auc_roc'],
                        'f1': val_stats['aggregated']['f1'], 
                        'bacc': val_stats['aggregated']['bacc'],
                        'dir_path': output_path
                    }, best_checkpoint_path
                )

        else: 
            # Single scale mil models 

            print(f"\nTrain Loss: {train_stats['loss']:.4f} | Train F1-Score: {train_stats['f1']:.4f} | Train Bacc: {train_stats['bacc']:.4f} | Train ROC-AUC: {train_stats['auc_roc']:.4f}")
            
            print(f"\nVal Loss: {val_stats['loss']:.4f} | Val F1-Score: {val_stats['f1']:.4f} | Val. Bacc: {val_stats['bacc']:.4f} | Val ROC-AUC: {val_stats['auc_roc']:.4f}\n")
        
            # Update results dictionary
            train_results['loss'].append(train_stats['loss'])
            train_results['f1'].append(train_stats['f1'])
            train_results['bacc'].append(train_stats['bacc'])
            train_results['auc_roc'].append(train_stats['auc_roc'])
            train_results['lr'].append(train_stats['lr'])
                
            val_results['loss'].append(val_stats['loss'])
            val_results['f1'].append(val_stats['f1'])
            val_results['bacc'].append(val_stats['bacc'])
            val_results['auc_roc'].append(val_stats['auc_roc'])

            # Save checkpoint if best validation ROC-AUC so far
            if best_aucroc < val_stats['auc_roc']:
                best_aucroc = val_stats['auc_roc']
                best_val_stats = val_stats 
    
                best_epoch = epoch + 1
                              
                model_name = 'best_model.pth'
                best_checkpoint_path = output_path / model_name
                
                print(f'Epoch {epoch + 1} - Save aucroc: {best_aucroc:.4f} Model')
                    
                torch.save(
                    { 
                        'model': model.state_dict(),
                        #'predictions': val_predictions,
                        'epoch': epoch,
                        'auroc': val_stats['auc_roc'],
                        'f1': val_stats['f1'], 
                        'bacc': val_stats['bacc'],
                        'dir_path': output_path
                    }, best_checkpoint_path
                )

        print(f'\nbest AUC-ROC Score at epoch {best_epoch}: {best_aucroc:.4f}')

    # Plot learning rate scheduler curve and training/validation metrics curves
    plot_lrs_scheduler(train_results['lr'], output_path)
    plot_loss_and_acc_curves(train_results, val_results, 'auc_roc', output_path)

    # Clear GPU memory cache and garbage collect
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_val_stats, best_checkpoint_path

def train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, device):
    """
    Training loop for one epoch.
    """
        
    model.train() # Set model to training mode
    model.is_training = True 
    
    losses = AverageMeter()

    progress_iter = tqdm(enumerate(train_loader), 
                         desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]",
                         total=len(train_loader)
                        )
    
    targs = []

    if args.mil_type == 'pyramidal_mil':
        preds = {}
        probs = {}

        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
            
            for s in args.scales: 
                preds[s] = []
                probs[s] = []
            
        preds['aggregated'] = []
        probs['aggregated'] = []

    else:
        preds = []
        probs = []
        
    start = time.time()

    # Iterate over batches
    for step, data in progress_iter:

        # Send data to device
        if isinstance(data['x'], dict): 
            inputs = {scale: tensor.to(device) for scale, tensor in data['x'].items()}
            batch_size = inputs[args.scales[0]].size(0)
        elif isinstance(data['x'], list): 
            inputs = [tensor.to(device) for tensor in data['x']]
            batch_size = inputs[0].size(0)
        else: 
            inputs = data['x'].to(device) 
            batch_size = inputs.size(0)

        labels = data['y'].float().to(device)
        
        # Wrap forward pass with autocast
        with torch.cuda.amp.autocast(enabled=args.apex):

            if args.mil_type == 'pyramidal_mil':
                if args.type_scale_aggregator in ['concatenation', 'gated-attention']:  

                    # Model returns logits for the scale-specific and multi-scale branches if deep supervision enabled
                    if args.deep_supervision: 
                        logits, side_logits = model(inputs) 
                    else: # Model returns logits for the multi-scale branch 
                        logits= model(inputs) 
                    
                    logits = logits.nan_to_num()
                    
                    loss = criterion(logits.view(-1, 1), labels.view(-1, 1))
                    
                elif args.type_scale_aggregator in ['max_p', 'mean_p']: 
                    side_logits = model(inputs)
                    
                    loss = 0.0 
            
            else: 
                # single-scale mil models 
                logits = model(inputs)

                loss = criterion(logits.view(-1, 1), labels.view(-1, 1))

        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
            
            for idx, side_logit in enumerate(side_logits): 
                side_logit = side_logit.nan_to_num()
                    
                loss += criterion(side_logit.view(-1, 1), labels.view(-1, 1))
        
        losses.update(loss.item(), batch_size)

        # Backprop w/ gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping if enabled
        if args.clip_grad > 0.0:
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # Step optimizer and update scaler
        scaler.step(optimizer)
        scaler.update() 
        
        optimizer.zero_grad() # Clear gradients for next step

        # Step learning rate scheduler per batch
        scheduler.step()

        targs.append(labels.cpu().numpy()) 

        
        if args.mil_type == 'pyramidal_mil': # store predictions and probabilities for multi-scale MIL models

            # Store predictions and probabilities depending on multi-scale aggregator 
            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 

                # Store scale-specific predictions and probabilities
                for idx, s in enumerate(args.scales): 
                    y_probs = side_logits[idx].sigmoid().detach()
                    y_probs = y_probs.nan_to_num()
                    
                    y_preds = (y_probs > 0.5).float()
                    
                    probs[s].append(y_probs.cpu().numpy())
                    preds[s].append(y_preds.cpu().numpy())

            # Store multi-scale aggregated predictions and probabilities depending on multi-scale aggregator 
            if args.type_scale_aggregator in ['concatenation', 'gated-attention']:
                y_probs = logits.sigmoid().detach()
                y_probs = y_probs.nan_to_num()
                
                y_preds = (y_probs > 0.5).float()
    
                probs['aggregated'].append(y_probs.cpu().numpy())
                preds['aggregated'].append(y_preds.cpu().numpy())
    
            elif args.type_scale_aggregator in ['max_p', 'mean_p']:
                # mean or max pooling over side logits 
                y_probs_aggregated = torch.zeros_like(y_probs)
    
                for idx, s in enumerate(args.scales): 
                    y_probs = side_logits[idx].sigmoid().detach()
                    y_probs = y_probs.nan_to_num()  # Ensure no NaNs
    
                    if args.type_scale_aggregator == 'mean_p': 
                        y_probs_aggregated += y_probs/len(args.scales)
                        
                    if args.type_scale_aggregator == 'max_p': 
                        y_probs_aggregated = torch.maximum(y_probs_aggregated, y_probs)
                        
                y_preds_aggregated = (y_probs_aggregated > 0.5).float()
                    
                probs['aggregated'].append(y_probs_aggregated.cpu().numpy())
                preds['aggregated'].append(y_preds_aggregated.cpu().numpy()) 
        
        else: # store predictions and probabilities for single-scale mil models 
            y_probs = logits.sigmoid().detach()
            y_preds = (y_probs > 0.5).float()
    
            probs.append(y_probs.cpu().numpy())
            preds.append(y_preds.cpu().numpy())

            
        progress_iter.set_postfix(
            {
                "lr": [optimizer.param_groups[0]['lr']],
                "loss": f"{losses.avg:.4f}",
                #"loss": f"{train_loss:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

    train_stats = {
        'loss': losses.avg, 
        'lr': optimizer.param_groups[0]['lr']
    }

    targs = np.concatenate(targs)

    # Compute and store metrics depending on MIL model type
    if args.mil_type == 'pyramidal_mil':

        # Metrics per scale if applicable
        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
            for s in args.scales:
                
                preds_s = np.concatenate(preds[s])
                probs_s = np.concatenate(probs[s])
        
                aucroc = auroc(targs, probs_s)
                f1, bacc = evaluate_metrics(targs, preds_s)
        
                train_stats[s] = {'auc_roc': aucroc, 'bacc': bacc, 'f1': f1}

        # Metrics on aggregated predictions
        preds = np.concatenate(preds['aggregated'])
        probs = np.concatenate(probs['aggregated'])

        aucroc = auroc(targs, probs)
        f1, bacc = evaluate_metrics(targs, preds) 
    
        train_stats['aggregated'] = {'auc_roc': aucroc, 'bacc': bacc, 'f1': f1}

    else: # single-scale mil models 
        preds = np.concatenate(preds)
        probs = np.concatenate(probs)
    
        aucroc = auroc(targs, probs)
        f1, bacc = evaluate_metrics(targs, preds)

        train_stats.update({'auc_roc': aucroc, 'bacc': bacc, 'f1': f1})
    
    return train_stats 

@torch.no_grad()
def valid_fn(valid_loader, model, criterion, args, device, split = 'val', epoch=1):
    
    model.eval() # Set model to evaluation mode
    model.is_training = False 
    
    losses = AverageMeter() 

    targs = []

    if args.mil_type == 'pyramidal_mil':
        preds = {}
        probs = {}

        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:
    
            for s in args.scales: 
                preds[s] = []
                probs[s] = []
            
        preds['aggregated'] = []
        probs['aggregated'] = []

    else: 
        preds = []
        probs = []
            
    start = time.time()

    if split == 'val': 
        progress_iter = tqdm(enumerate(valid_loader), 
                             desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]",
                             total=len(valid_loader)
                            )
    else:
        progress_iter = tqdm(enumerate(valid_loader), 
                             total=len(valid_loader)
                            )
    
    for step, data in progress_iter:

        # Send data to device
        if isinstance(data['x'], dict): 
            inputs = {scale: tensor.to(device, non_blocking=True) for scale, tensor in data['x'].items()}
            batch_size = inputs[args.scales[0]].size(0)
        elif isinstance(data['x'], list): 
            inputs = [tensor.to(device, non_blocking=True) for tensor in data['x']]
            batch_size = inputs[0].size(0)
        else: 
            inputs = data['x'].to(device, non_blocking=True)
            batch_size = inputs.size(0)

        labels = data['y'].float().to(device)
        
        # Wrap forward pass with autocast
        with torch.cuda.amp.autocast(enabled=args.apex):

            if args.mil_type == 'pyramidal_mil': 
                
                if args.type_scale_aggregator in ['concatenation', 'gated-attention']:

                    # Model returns logits for the scale-specific and multi-scale branches if deep supervision enabled
                    if args.deep_supervision: 
                        logits, side_logits = model(inputs) 
                    else: # Model returns logits only for the multi-scale branch 
                        logits = model(inputs) 
                        
                    logits = logits.nan_to_num()
                    
                    loss = criterion(logits.view(-1, 1), labels.view(-1, 1))
                    
                elif args.type_scale_aggregator in ['mean_p', 'max_p']:
                    side_logits = model(inputs)
                    
                    loss = 0.0 

            else: # single-scale mil models 
                logits = model(inputs)

                loss = criterion(logits.view(-1, 1), labels.view(-1, 1))

        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:
            for idx, side_logit in enumerate(side_logits): 
                side_logit = side_logit.nan_to_num()
                    
                loss += criterion(side_logit.view(-1, 1), labels.view(-1, 1))
                
        losses.update(loss.item(), batch_size)

        targs.append(labels.cpu().numpy()) 

        if args.mil_type == 'pyramidal_mil':

            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:
                
                # Store scale-specific predictions and probabilities 
                for idx, s in enumerate(args.scales): 
                    y_probs = side_logits[idx].sigmoid().detach()
                    y_probs = y_probs.nan_to_num()
                    
                    y_preds = (y_probs > 0.5).float()
                    
                    probs[s].append(y_probs.cpu().numpy())
                    preds[s].append(y_preds.cpu().numpy())

            # store multi-scale aggregated probabilities and predictions depending on multi-scale aggregator type 
            if args.type_scale_aggregator in ['concatenation', 'gated-attention']:
                y_probs = logits.sigmoid().detach()
                y_probs = y_probs.nan_to_num()
                
                y_preds = (y_probs > 0.5).float()
    
                probs['aggregated'].append(y_probs.cpu().numpy())
                preds['aggregated'].append(y_preds.cpu().numpy())
    
            elif args.type_scale_aggregator in ['mean_p', 'max_p']:
                # multi-scale aggregated results --> mean or max pooling over scale-specific probabilities and predictions 
                y_probs_aggregated = torch.zeros_like(y_probs)
    
                for idx, s in enumerate(args.scales): 
                    y_probs = side_logits[idx].sigmoid().detach()
                    y_probs = y_probs.nan_to_num()
    
                    if args.type_scale_aggregator == 'mean_p': 
                        y_probs_aggregated += y_probs/len(args.scales)
                        
                    if args.type_scale_aggregator == 'max_p': 
                        y_probs_aggregated = torch.maximum(y_probs_aggregated, y_probs)
                        
                y_preds_aggregated = (y_probs_aggregated > 0.5).float()
                    
                probs['aggregated'].append(y_probs_aggregated.cpu().numpy())
                preds['aggregated'].append(y_preds_aggregated.cpu().numpy()) 

        else: # store predictions and probabilities for single-scale mil models 

            y_probs = logits.sigmoid().detach()
            y_preds = (y_probs > 0.5).float()
    
            probs.append(y_probs.cpu().numpy())
            preds.append(y_preds.cpu().numpy())
        
        progress_iter.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

    val_stats = {
        'loss': losses.avg, 
    }

    targs = np.concatenate(targs)

    # Compute and store metrics depending on MIL model type
    if args.mil_type == 'pyramidal_mil':

        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:

            # Metrics per scale if applicable
            for s in args.scales:
    
                preds_s = np.concatenate(preds[s])
                probs_s = np.concatenate(probs[s])
        
                aucroc = auroc(targs, probs_s)
                f1, bacc = evaluate_metrics(targs, preds_s)
                
                val_stats[s] = {'auc_roc': aucroc, 'bacc': bacc, 'f1': f1}

        # Metrics on aggregated predictions
        preds = np.concatenate(preds['aggregated'])
        probs = np.concatenate(probs['aggregated'])
    
        aucroc = auroc(targs, probs)
        f1, bacc = evaluate_metrics(targs, preds) 
        cf_matrix = confusion_matrix(targs, preds) if split == 'test' else None
            
        val_stats['aggregated'] = {'auc_roc': aucroc, 'bacc': bacc, 'f1': f1, 'cf_matrix': cf_matrix}

    else: # single-scale mil models

        preds = np.concatenate(preds)
        probs = np.concatenate(probs)
    
        aucroc = auroc(targs, probs)
        f1, bacc = evaluate_metrics(targs, preds) 
        cf_matrix = confusion_matrix(targs, preds) if split == 'test' else None
        
        val_stats.update({'auc_roc': aucroc, 'bacc': bacc, 'f1': f1, 'cf_matrix': cf_matrix})
    
    if split == 'test': 
        return targs, preds, probs, val_stats

    return val_stats 
