import numpy as np
import pandas as pd
import pickle
import torch
import torchvision
import torchvision.transforms
import torch.nn.functional as F
from albumentations import *
from torch.utils.data import DataLoader
import cv2

from .dataset_concepts import Generic_MIL_Dataset_Detection, collate_MIL_patches_detection, Generic_MIL_Dataset, collate_MIL_patches, BagDataset, collate_patch_features

from utils.generic_utils import clear_memory

class center_crop(object):
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img):
        return self.crop_center(img)


class AlbumentationsTransform:
    """
    Wrapper for applying an Albumentations transform to an image.
    
    Args:
        transform (callable): Albumentations transform object.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img_array):

        augmented = self.transform(img_array)
        return augmented

class lambda_funct(torchvision.transforms.Lambda):
    """
    Lambda transform wrapper for padding an image.

    Args:
        lambd (callable): Padding function to apply.
        patch_size (int): Target patch size to pad to.
        mean (float): Mean for normalization.
        std (float): Std for normalization.
    """
    def __init__(self, lambd, patch_size, mean, std):
        super().__init__(lambda_funct)
        
        self.lambd = lambd
        self.patch_size = patch_size 
        self.mean = mean
        self.std = std 

    def __call__(self, img):

        return self.lambd(img, self.patch_size, self.mean, self.std)

class lambda_funct_rot(torchvision.transforms.Lambda):
    """
    Lambda transform for applying random rotation to an image.

    Args:
        lambd (callable): Rotation function.
        angles (list): List of angles to choose from.
    """
    def __init__(self, lambd, angles):
        super().__init__(lambda_funct_rot)
        
        self.lambd = lambd
        self.angles = angles

    def __call__(self, img):

        return self.lambd(img, self.angles)
        
def RotationTrans(x, angles):
    """Rotate by one of the given angles."""

    angle = random.choice(angles)

    return torchvision.transforms.functional.rotate(x, angle)


def get_transforms(args):
    """
    Returns a composition of data augmentation transforms used for training.
    """
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        lambda_funct_rot(RotationTrans, [0, 90, 180, 270]),  
        #transforms.ColorJitter(0.25, 0.25, 0.25, 0.25), 
        #transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))
    ])

    
def pad_image(img_array, patch_size, mean, std):
    """
    Pads an image tensor so that its height and width are multiples of the patch size.

    Args:
        img_array (Tensor): Input image tensor with shape (C, H, W).
        patch_size (int): patch size.
        mean (float): Mean used for normalization.
        std (float): Standard deviation used for normalization.

    Returns:
        padded_img (Tensor): Padded image tensor.
        padding (tuple): Tuple of applied paddings: (left, right, top, bottom).
    """

    # Get dimensions of the image
    if len(img_array.size()) == 3: # If image has channel dimension
        c, h, w = img_array.size()
    else: # Just height and width
        h, w = img_array.size()

    # Compute new dimensions that are divisible by patch_size
    new_h = h + (patch_size - h%patch_size)
    new_w = w + (patch_size - w%patch_size)
        
    # Determine needed padding for width and height
    additional_h = new_h - h
    additional_w = new_w - w
        
    # Initialize padding amounts
    padding_left, padding_right, padding_top, padding_bottom = 0, 0, 0, 0
        
    # Horizontal sum (sums over height)
    horizontal_sum = img_array.sum(axis=(0, 1))
    left_info = horizontal_sum[:w//2].sum()
    right_info = horizontal_sum[w//2:].sum()

    # Apply padding on the side with less information for width
    if left_info < right_info:
        padding_left = additional_w
    else:
        padding_right = additional_w

    # Vertical sum (sums over width)
    vertical_sum = img_array.sum(axis=(0, 2))
    top_info = vertical_sum[:h//2].sum()
    bottom_info = vertical_sum[h//2:].sum()
    
    # Apply padding on the side with less information for height
    if top_info < bottom_info:
        padding_top = additional_h
    else:
        padding_bottom = additional_h

    # Construct the padding configuration
    normalized_black_value = (0.0 - mean) / std # Compute padding value (normalized black)
    padded_img = F.pad(img_array, 
                       (padding_left, padding_right, padding_top, padding_bottom),
                       mode='constant', 
                       value = normalized_black_value
                      )
    return padded_img, (padding_left, padding_right, padding_top, padding_bottom)


class Patching:
    """
    Extracts patches from an image. 

    Args:
        patch_size (int): Default patch size (used if multi_scale_model is None).
        overlap (float or list): Overlap between patches.
        multi_scale_model (str or None): One of ['msp', 'fpn', 'backbone_pyramid'], or None for single scale.
        scales (list): List of scales to use if multi_scale_model == 'msp'.
        mean (float): Mean pixel value used for normalization (for padding).
        std (float): Standard deviation used for normalization (for padding).
    """
    
    def __init__(self, patch_size=512, overlap=0, multi_scale_model=None, scales=[16, 8, 4], mean = 0.3089279, std = 0.25053555408335154):
        self.patch_size = patch_size if multi_scale_model is not None else scales[0]
        self.overlap = overlap 
        self.multi_scale_model = multi_scale_model
        self.scales = scales

        self.mean = mean 
        self.std = std 

    def extract_patch(self, image, x_start, y_start, size, img_h, img_w):
        """
        Extracts a single patch from the image and pads it if it extends beyond image boundaries.
        """
        
        # Define patch bounds
        x_end = x_start + size
        y_end = y_start + size
    
        # Compute the effective patch size
        x_pad_start = max(0, -x_start)
        y_pad_start = max(0, -y_start)
        x_pad_end = max(0, x_end - img_w)
        y_pad_end = max(0, y_end - img_h)
    
        # Ensure the starting and ending positions are within the image boundaries
        x_start_clipped = max(0, x_start)
        y_start_clipped = max(0, y_start)
        x_end_clipped = min(img_w, x_end)
        y_end_clipped = min(img_h, y_end)

        # Extract the valid region of the patch from the image
        patch = image[:, y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]

        # Normalize padding value (black pixel normalized)
        normalized_black_value = (0 - self.mean) / self.std

        # Pad to match patch size if needed
        patch = F.pad(
            patch,
            pad=(x_pad_start, x_pad_end, y_pad_start, y_pad_end),
            mode='constant',
            value = normalized_black_value 
        )
        
        return patch, x_start, y_start

    
    def __call__(self, img_array_with_padding):

        # Unpack the input
        img_array, padding = img_array_with_padding
        c, img_height, img_width = img_array.shape

        # Initialize dictionaries for patches and coordinates
        if self.multi_scale_model == 'msp':
            patches = {size: [] for size in self.scales}
            patch_coords = {size: [] for size in self.scales}
        else:
            patches = []
            patch_coords = []

        if self.multi_scale_model == 'msp': 
            # Multi-scale patching 
            for idx, patch_size in enumerate(self.scales): 
                step_size = patch_size - int(patch_size * self.overlap[idx])
                #step_size = 64
                
                start_x, start_y, w, h = (0, 0, img_width, img_height)
                x_range = range(0, img_width, step_size)
                y_range = range(0, img_height, step_size)
                
                for x in x_range:
                    for y in y_range:
                        patch, x_start, y_start = self.extract_patch(img_array, x, y, patch_size, h, w)
                        
                        patches[patch_size].append(patch)
                        patch_coords[patch_size].append([int(x_start), int(y_start)])
    
        elif self.multi_scale_model in ['fpn', 'backbone_pyramid']: 
            step_size = self.patch_size - int(self.patch_size * self.overlap[0])
            
            start_x, start_y, w, h = (0, 0, img_width, img_height)
            stop_y = min(start_y + h, h - self.patch_size + 1)
            stop_x = min(start_x + w, w - self.patch_size + 1)
            x_range = np.arange(start_x, stop_x, step=step_size)
            y_range = np.arange(start_y, stop_y, step=step_size)

            for x in x_range:
                for y in y_range:
                    patch = img_array[:, y:y + self.patch_size, x:x + self.patch_size]

                    patches.append(patch)
                    patch_coords.append([int(x), int(y)])

        else: 
            # Standard single-scale patching
            patch_size = self.scales[0]
            step_size = patch_size - int(patch_size * self.overlap[0])
                
            start_x, start_y, w, h = (0, 0, img_width, img_height)
            x_range = range(0, img_width, step_size)
            y_range = range(0, img_height, step_size)
                
            for x in x_range:
                for y in y_range:
                    patch, x_start, y_start = self.extract_patch(img_array, x, y, patch_size, h, w)
                        
                    patches.append(patch)
                    patch_coords.append([int(x_start), int(y_start)])

            
        # Convert lists to tensors or arrays
        if self.multi_scale_model == 'msp': 
            patch_coords = {size: np.array(patch_coords[size]) for size in patch_coords}
            patches = {size: torch.stack(patches[size]) for size in patches}
            
        else:
            patch_coords = np.array(patch_coords)
            patches = torch.stack(patches)

        # Sort patch coordinates and patches
        if isinstance(patch_coords, dict):  # Multi-scale image pyramid
            for size in patch_coords:
                sorted_indices = np.lexsort((patch_coords[size][:, 0], patch_coords[size][:, 1]))  # Sort by y, then x
                patch_coords[size] = patch_coords[size][sorted_indices]
                patches[size] = patches[size][sorted_indices]
        else:
            sorted_indices = np.lexsort((patch_coords[:, 0], patch_coords[:, 1]))  # Sort by y, then x
            patch_coords = patch_coords[sorted_indices]
            patches = patches[sorted_indices]

        return patches, patch_coords, padding

def bags_dataloader(df, args):

    tfm = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize(mean=args.mean, std=args.std),
        lambda_funct(pad_image, args.patch_size, args.mean, args.std),
        Patching(patch_size = args.patch_size, overlap = args.overlap, multi_scale_model = args.multi_scale_model, scales = args.scales)
    ])

    dataset = BagDataset(args=args, df=df, transform=tfm)      

    loader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True,
            drop_last=False, 
            collate_fn=collate_patch_features 
        )

    return loader 


def MIL_dataloader(split_df, split, args):
    """
    Creates and returns a PyTorch DataLoader for MIL tasks, handling both online feature extraction and pre-extracted features, as well as optional ROI evaluation and data augmentation.

    Args:
        split_df (pd.DataFrame): DataFrame containing data for the given split (train/val/test).
        split (str): 'train', 'val', or 'test'.
        args (Namespace): Parsed arguments containing configuration settings.

    Returns:
        DataLoader: Configured DataLoader for the specified split and settings.
    """

    # Define Transformations
    if args.feature_extraction == 'online': 
        # Online feature extraction 
        if split == 'train' and args.data_aug: 
             # if data augmentation, applied only during training
            tfm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  AlbumentationsTransform(get_transforms(args)), # Custom augmentations
                                                  torchvision.transforms.Normalize(mean=args.mean, std=args.std),
                                                  lambda_funct(pad_image, args.patch_size, args.mean, args.std), # Pad to patch size
                                                  Patching(
                                                      patch_size = args.patch_size, 
                                                      overlap = args.overlap, 
                                                      multi_scale_model = args.multi_scale_model, 
                                                      scales = args.scales
                                                  )
                                                 ])
        else: 
            # No augmentation for validation/test; and also for training if args.data_aug = False
            tfm = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=args.mean, std=args.std),
                                                  lambda_funct(pad_image, args.patch_size, args.mean, args.std),              
                                                  Patching(
                                                      patch_size = args.patch_size, 
                                                      overlap = args.overlap, 
                                                      multi_scale_model = args.multi_scale_model, 
                                                      scales = args.scales
                                                  )
                                                 ])
        
    else:     
        # Use pre-extracted features (no image transforms needed) 
        tfm = None
        
    if args.roi_eval: 
        # Use ROI-specific dataset for detection evaluation
        split_dataset = Generic_MIL_Dataset_Detection(args=args, df=split_df, transform=tfm)

    else: 
        # Standard MIL dataset
        split_dataset = Generic_MIL_Dataset(args=args, df=split_df, transform=tfm) 

    # DataLoader Configuration
    if split == 'train':

        loader = DataLoader(
            split_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_MIL_patches 
        )
        
    else: 
                
        loader = DataLoader(
            split_dataset, 
            batch_size=args.batch_size if not args.roi_eval else 1, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True,
            drop_last=False, 
            collate_fn=collate_MIL_patches_detection if args.roi_eval else collate_MIL_patches
        )

    return loader 

