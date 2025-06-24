#internal imports 
#from breastclip.scheduler import LinearWarmupCosineAnnealingLR
from MIL.MILmodels import EmbeddingMIL, PyramidalMILmodel, NestedPyramidalMILmodel

#external imports 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Union
from torch.optim import Optimizer

class Training_Stage_Config:
    """
    Configures the training mode of a model for different training strategies (frozen, finetune, finetune with warmup).
    
    Args:
    - model (torch.nn.Module): The model to configure (should be of type EmbeddingMIL or PyramidalMIL).
    - training_mode (str): The training strategy ('frozen', 'finetune', 'finetune_warmup').
    - warmup_epochs (int): Number of epochs for warmup in 'finetune_warmup' mode.
    """
    def __init__(self, model: torch.nn.Module, training_mode: str, warmup_epochs: int):

        model.train() # Ensure model is in training mode
        
        self.warmup_epochs = warmup_epochs
        self.training_mode = training_mode 

        if training_mode == 'frozen': 
            print(f"[INFO] - instance encoder is frozen during training.")
            if isinstance(model, EmbeddingMIL):
                self._freeze_parameters(model.inst_encoder)
                
            elif isinstance(model, PyramidalMILmodel) or isinstance(model,NestedPyramidalMILmodel):
                self._freeze_parameters(model.inst_encoder.backbone) 

        elif training_mode == 'finetune': 
            if warmup_epochs > 0:
                print(f"[INFO] - Warmup phase: instance encoder is frozen.")
                if isinstance(model, EmbeddingMIL):
                    self._freeze_parameters(model.inst_encoder)
                elif isinstance(model, PyramidalMILmodel) or isinstance(model,NestedPyramidalMILmodel):
                    self._freeze_parameters(model.inst_encoder.backbone) 

            else: 
                print(f"[INFO]: Finetune phase: Unfreeze top layers from the instance encoder")
                if isinstance(model, EmbeddingMIL):
                    self._freeze_parameters(model.inst_encoder)
                    self._unfreeze_top_layers(model.inst_encoder)
                        
                elif isinstance(model, PyramidalMILmodel) or isinstance(model,NestedPyramidalMILmodel):
                    self._freeze_parameters(model.inst_encoder.backbone)
                    self._unfreeze_top_layers(model.inst_encoder.backbone)
                    
    def _freeze_parameters(self, module):
        """Helper function to freeze parameters."""

        #module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_parameters(self, module):
        """Helper function to unfreeze parameters."""

        #module.train()
        
        for param in module.parameters():
            param.requires_grad = True

    def _unfreeze_top_layers(self, module, optimizer = None, current_lr = 0.0, add_param_group = False):
        """
        Unfreeze the top layers of the given module.
        
        Args:
            module (torch.nn.Module): The module containing layers (e.g., encoder.backbone).
            optimizer (torch.optim.Optimizer, optional): Optimizer to add new param groups.
            current_lr (float): Current base learning rate.
            add_param_group (bool): If True, adds top layer params to optimizer with 0.1 * lr.
        """

        for block_num in range(1, 9): 
            block = module._blocks[-block_num]  
            
            for param in block.parameters():   
                param.requires_grad = True

            # If specified, add block to optimizer with reduced learning rate
            if add_param_group: 
                optimizer.add_param_group({'params': block.parameters(), 'lr': current_lr*0.1}) 
     
    
    def __call__(self, model, optimizer, current_epoch, current_lr): 

        """
        Callable stage manager. Transitions from warmup to finetune by unfreezing top layers.

        Args:
            model (torch.nn.Module): The current model.
            optimizer (torch.optim.Optimizer): The optimizer to update.
            current_epoch (int): Current training epoch.
            current_lr (float): Current learning rate.
        """

        if self.training_mode == 'finetune': 
            # When warmup ends, start unfreezing top layers
            if current_epoch == self.warmup_epochs and current_epoch > 0:
                print(f"[INFO]: Finetune phase: Unfreeze top layers from the instance encoder")
                if isinstance(model, EmbeddingMIL):
                    self._unfreeze_top_layers(model.inst_encoder, optimizer, current_lr, add_param_group = True)
                    
                elif isinstance(model, PyramidalMILmodel) or isinstance(model,NestedPyramidalMILmodel):
                    self._unfreeze_top_layers(model.inst_encoder.backbone, optimizer, current_lr, add_param_group = True)


class LinearWarmupCosineAnnealingLR(LambdaLR):
    r"""
    A learning rate scheduler which linearly increases learning rate from 0
    LR, and further decreases it to zero by cosine decay. After linear warmup,
    the LR decays as:
    .. math::
        \eta_t = \eta_{max}\cos^2(\frac{T_{cur} - T_{warm}}{T_{max} - T_{warm}}\frac{\pi}{2})
    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Wrapper optimizer.
    total_steps: int
        Total epochs (or iterations) for training.
    warmup_steps: int
        Number of first few steps to do linear warmup.
    last_epoch: int, optional (default = -1)
        The index of last step (epoch or iteration). We named it ``last_epoch``
        instead of ``last_step`` to keep the naming consistent with other LR
        schedulers in PyTorch.
    """

    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: Union[int, float], last_epoch: int = -1,
                 **kwargs):
        assert warmup_steps < total_steps, "Warmup steps should be less than total steps."

        self.tsteps = total_steps
        if isinstance(warmup_steps, float):
            self.wsteps = math.ceil(total_steps * warmup_steps)
        else:
            self.wsteps = warmup_steps

        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.wsteps:
            # Linear warmup.
            multiplier = step / float(max(1, self.wsteps))
        else:
            # Cosine annealing decay.
            cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
            multiplier = math.cos(cos_factor * (math.pi / 2)) ** 2
        # Avoid negative learning rate.
        return max(0, multiplier)


def initialize_training_setup(train_loader, model, device, args):
    """
    Initialize optimizer, scheduler, scaler, and loss functions for training.
    """
    
    optimizer = None
    scheduler = None
    scaler = None

    # Optimizer Setup
    if args.training_mode == 'finetune' and args.warmup_stage_epochs == 0: 
        # Separate learning rates for backbone and head
        param_dicts = []

        param_dicts += [
            {
                "params": [p for n, p in model.named_parameters() if ("backbone" not in n) and (p.requires_grad)],
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr*0.1,
            },
        ]
    
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        
    else: 
        # Same learning rate for all trainable parameters
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # Learning Rate Scheduler 
    if args.warmup_epochs == 0.1:
        warmup_steps = args.epochs
    elif args.warmup_epochs == 1:
        warmup_steps = len(train_loader)
    else:
        warmup_steps = 10
    lr_config = {
        'total_epochs': args.epochs,
        'warmup_steps': warmup_steps,
        'total_steps': len(train_loader) * args.epochs
    }
    
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)

    #AMP Gradient Scaler for Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler()

    # Loss Function Setup
    pos_wt = torch.tensor([args.BCE_weights]).to('cuda') 
    print(f'pos_wt: {pos_wt}') 

    train_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight =pos_wt) if args.weighted_BCE == "y" else torch.nn.BCEWithLogitsLoss()
    eval_criterion = train_criterion 

    return optimizer, scheduler, scaler, train_criterion, eval_criterion