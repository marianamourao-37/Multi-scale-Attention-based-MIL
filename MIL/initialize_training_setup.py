from breastclip.scheduler import LinearWarmupCosineAnnealingLR
from breastclip.models.breast_clip_classifier import BreastClipClassifier
from .MILmodels import EmbeddingMIL, PyramidalMILmodel


#external imports 
import torch 
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, sample_weights = None, reduction='mean', device = None):
        """
        Focal Loss for classification.
        
        :param gamma: focusing parameter that adjusts how the modulating factor is applied
        :param alpha: weight balancing factor for classes
        :param reduction: 'mean' or 'sum' or 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.sample_weights = sample_weights
        self.reduction = reduction
        self.device = device 

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (logits), expected shape [batch_size, n_classes]
            target: Ground truth labels, same shape as pred
        """
        
        # Get the probabilities by applying the sigmoid activation
        pred_sigmoid = pred.sigmoid()
        pred_sigmoid = torch.clamp(pred_sigmoid, 1e-4, 1.0 - 1e-4)
        
        # Ensure target is same type as predictions
        target = target.type_as(pred)
        
        # Compute pt
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)

        # Dynamically assign gamma based on the probability range of pred_sigmoid (P(Y_j = 1))
        gamma = torch.ones_like(pred_sigmoid)  # Start with a tensor of ones (shape [batch_size, n_classes])
        
        # Set gamma = 5 where pred_sigmoid is in [0, 0.2), gamma = 3 otherwise
        gamma[pred_sigmoid < 0.2] = 5.0
        gamma[pred_sigmoid >= 0.2] = 3.0
        
        # Compute the focal loss
        #F_loss = pt.pow(self.gamma) * F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        F_loss = pt.pow(gamma) * F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # If alpha is provided, apply the class balancing factor
        if self.alpha:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            F_loss = alpha_t * F_loss

        if self.sample_weights is not None: 
            focal_sample_weights = torch.ones(target.shape).to(self.device)
            focal_sample_weights = torch.where(target == 1.0, self.sample_weights[1], self.sample_weights[0])
            
            #weights = self.sample_weights.view(F_loss.size(0), -1)
            #print('F_loss.shape:', F_loss.shape) 
            #print('focal_sample_weights.shape:', focal_sample_weights.shape) 
            
            F_loss *= focal_sample_weights

        # Apply the reduction method
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class Training_Stage_Config:
    """
    Configures the training mode of a model for different training strategies (frozen, finetune, finetune with warmup).
    
    Args:
    - model (torch.nn.Module): The model to configure (should be of type EmbeddingMIL or PyramidalMIL).
    - training_mode (str): The training strategy ('frozen', 'finetune', 'finetune_warmup').
    - warmup_epochs (int): Number of epochs for warmup in 'finetune_warmup' mode.
    """
    def __init__(self, model: torch.nn.Module, training_mode: str, warmup_epochs: int):

        self.warmup_epochs = warmup_epochs

        self.training_mode = training_mode 

        if training_mode == 'frozen': 
            print(f"[INFO] - instance encoder is frozen during training.")
            if isinstance(model, EmbeddingMIL):
                self._freeze_parameters(model.inst_encoder)
                
            elif isinstance(model, PyramidalMILmodel):
                self._freeze_parameters(model.inst_encoder.backbone) 

            elif isinstance(model, BreastClipClassifier): 
                self._freeze_parameters(model.image_encoder)

        elif training_mode == 'finetune': 
            if warmup_epochs > 0:
                print(f"[INFO] - Warmup phase: instance encoder is frozen.")
                if isinstance(model, EmbeddingMIL):
                    self._freeze_parameters(model.inst_encoder)
                elif isinstance(model, PyramidalMILmodel):
                    self._freeze_parameters(model.inst_encoder.backbone) 

                elif isinstance(model, BreastClipClassifier): 
                    self._freeze_parameters(model.image_encoder)
            else: 
                print(f"[INFO]: Finetune phase: Unfreeze top layers from the instance encoder")
                if isinstance(model, EmbeddingMIL):
                    self._freeze_parameters(model.inst_encoder.image_encoder)
                    self._unfreeze_top_layers(model.inst_encoder.image_encoder)
                        
                elif isinstance(model, PyramidalMILmodel):
                    self._freeze_parameters(model.inst_encoder.backbone)
                    self._unfreeze_top_layers(model.inst_encoder.backbone)

                elif isinstance(model, BreastClipClassifier): 
                    self._freeze_parameters(model.image_encoder)
                    self._unfreeze_top_layers(model.image_encoder)
                    
    def _freeze_parameters(self, module):
        """Helper function to freeze parameters."""
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_parameters(self, module):
        """Helper function to unfreeze parameters."""
        for param in module.parameters():
            param.requires_grad = True

    def _unfreeze_top_layers(self, module, optimizer = None, current_lr = 0.0, add_param_group = False):
        """
        Unfreeze the top `num_layers` layers of the given module.
        
        Args:
        - module (torch.nn.Module): The module containing layers to be unfrozen.
        - num_layers (int): Number of top layers to unfreeze.
        """

        #for param in module._blocks[-1].parameters(): 
        #    param.requires_grad = True

        #for param in module._blocks[-2].parameters(): 
        #    param.requires_grad = True

        for block_num in range(1, 9): 
            #print('block_num ', -block_num)
            
            for param in module._blocks[-block_num].parameters(): 
                param.requires_grad = True
                
        if add_param_group: 
            optimizer.add_param_group({'params': module._blocks[-1].parameters(), 'lr': current_lr*0.1}) 
     
    
    def __call__(self, model, optimizer, current_epoch, current_lr): 

        if self.training_mode == 'finetune': 
            if current_epoch == self.warmup_epochs and current_epoch > 0:
                print(f"[INFO]: Finetune phase: Unfreeze top layers from the instance encoder")
                if isinstance(model, EmbeddingMIL):
                    self._unfreeze_top_layers(model.inst_encoder, optimizer, current_lr)
                    
                elif isinstance(model, PyramidalMILmodel):
                    self._unfreeze_top_layers(model.inst_encoder.backbone, optimizer, current_lr)

                elif isinstance(model, BreastClipClassifier): 
                    self._unfreeze_top_layers(model.image_encoder, optimizer, current_lr)

def initialize_training_setup(train_loader, model, device, args):
    
    optimizer = None
    scheduler = None
    scaler = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
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
    scaler = torch.cuda.amp.GradScaler()

    pos_wt = torch.tensor([args.BCE_weights]).to('cuda') 
    print(f'pos_wt: {pos_wt}') 

    train_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight =pos_wt) if args.weighted_BCE == "y" else torch.nn.BCEWithLogitsLoss()
    eval_criterion = train_criterion

    return optimizer, scheduler, scaler, train_criterion, eval_criterion