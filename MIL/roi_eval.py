# internal imports
from Datasets.dataset_utils import MIL_dataloader, Generic_MIL_Dataset, collate_MIL_patches
from MIL import build_model 
from MIL.MIL_experiment import valid_fn
from utils.generic_utils import seed_all, clear_memory, print_network 
from utils.data_split_utils import stratified_train_val_split

# external imports 
from torch.utils.data import DataLoader

from pathlib import Path

import os 

from PIL import Image

from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

import torch 
from torchvision import transforms
import torch.nn.functional as F
from torchvision.ops import nms

import numpy as np 

import math

from tqdm import tqdm #progress bar

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import cv2

import pandas as pd

import contextlib
import io

from scipy import ndimage
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, median_filter, gaussian_filter
    
def pad_image(img_array, patch_size):
    """
    Pads an image tensor so that its height and width are divisible by the given patch_size.

    Parameters:
        img_array (Tensor): Input image as a PyTorch tensor (C x H x W) or (H x W).
        patch_size (int): The size of the patch to which the image dimensions must be aligned.

    Returns:
        padded_img (Tensor): Padded image tensor.
        padding_info (tuple): Tuple of padding values (left, right, top, bottom).
    """

    if len(img_array.size()) == 3: 
        c, h, w = img_array.size()
        h_axis = (0, 1)
        v_axis = (0, 2)
    else:
        h, w = img_array.size()
        h_axis = (0,)
        v_axis = (1,)

    # Compute new dimensions that are divisible by patch_size
    new_h = h + (patch_size - h%patch_size)
    new_w = w + (patch_size - w%patch_size)
        
    # Determine needed padding for width and height
    additional_h = new_h - h
    additional_w = new_w - w
        
    # Initialize padding amounts
    padding_left, padding_right, padding_top, padding_bottom = 0, 0, 0, 0

    # Horizontal sum (sums over height)
    horizontal_sum = img_array.sum(axis=h_axis)
    left_info = horizontal_sum[:w//2].sum()
    right_info = horizontal_sum[w//2:].sum()
    
    # Vertical sum (sums over width)
    vertical_sum = img_array.sum(axis=v_axis)
    top_info = vertical_sum[:h//2].sum()
    bottom_info = vertical_sum[h//2:].sum()

    # Apply padding on the side with less information for width
    if left_info < right_info:
        padding_left = additional_w
    else:
        padding_right = additional_w
    
    # Apply padding on the side with less information for height
    if top_info < bottom_info:
        padding_top = additional_h
    else:
        padding_bottom = additional_h

    # Construct the padding configuration
    padded_img = F.pad(img_array, 
                       (padding_left, padding_right, padding_top, padding_bottom),
                       mode='constant', 
                       value = img_array.min()
                      )
    
    return padded_img, (padding_left, padding_right, padding_top, padding_bottom)
    
def Get_Predicted_Class(label, predicted_class, label_type):

    if label_type.lower() == 'mass': 
        if label == 0:
            prefix = 'not mass'
        elif label == 1:
            prefix = 'mass'
        else:
            prefix = ''
            
        pred_class = 'not mass' if predicted_class < 0.5 else 'mass'

    if label_type.lower() == 'suspicious_calcification': 
        if label == 0:
            prefix = 'not calcification'
        elif label == 1:
            prefix = 'calcification'
        else:
            prefix = ''
        
        pred_class = 'not calcification' if predicted_class < 0.5 else 'calcification'
    
    return f'{prefix} | Pred: {pred_class}'
    
def Segment(image, sthresh=20, sthresh_up = 255, mthresh=7, close = 4, use_otsu=True):
    """
    Perform tissue segmentation on an input image using median filtering, followed by binary thresholding (Otsu or fixed) and optional morphological operations
    """

    image = image.cpu().numpy()
    image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)

    img_med = cv2.medianBlur(image, mthresh)  # Apply median blurring
    
    # Thresholding
    if use_otsu:
        _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    else:
        _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

    # Morphological closing
    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    # Convert back to float32 and normalize to [0, 1]
    img_otsu = img_otsu.astype(np.float32) / 255.0
    
    return torch.from_numpy(img_otsu)
    
def plot_image_with_boxes(image, 
                          boxes, 
                          mode = 'annotations', 
                          ax = None, 
                          cmap=plt.cm.bone):
    """
    Plot an image with bounding boxes.
    
    Args:
        image (ndarray): Input image.
        boxes (list): List of detections in the format [x1, y1, x2, y2] for ground-truth annotations or [x1, y1, x2, y2, score] for predictions.
        mode (str): 'annotations' or 'pred'. Determines how to interpret the box format.
        ax (matplotlib.axes.Axes): Axis to plot the image on.
        cmap (Colormap): Colormap to use for the image.
    """
    ax.imshow(image, cmap=cmap)

    if boxes is not None: 
        for box in boxes:
            if mode == 'pred':  
                xmin, ymin, xmax, ymax, score = box
            else: 
                xmin, ymin, xmax, ymax = box
                score = None 
                
            rect = patches.Rectangle(
                (xmin, ymin), 
                xmax - xmin, 
                ymax - ymin,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
                label=f"Score: {score:.2f}" if score is not None else None,
            )
            ax.add_patch(rect)


def ShowVis(heatmap, img, predicted_bboxes, axs, args):
    """
    Overlay a heatmap on an image and visualize it with predicted bounding boxes (if available).

    Args:
        heatmap (Tensor): Heatmap of shape (H, W), normalized to [0, 1].
        img (ndarray): Original RGB image as a float32 array in range [0, 1].
        predicted_bboxes (list): List of predicted boxes [x1, y1, x2, y2, score].
        axs (matplotlib.axes.Axes): Axis object for plotting.
        args (Namespace): Parsed arguments containing configuration settings. 
    """

    heatmap = cv2.applyColorMap(np.uint8(heatmap.numpy()*255), cv2.COLORMAP_JET)  
    heatmap = np.float32(heatmap) / 255

    # Overlay heatmap on the original image
    cam = heatmap*0.9 + np.float32(img)
    cam = cam / np.max(cam)

    # Convert overlay back to uint8
    vis =  np.uint8(255 * cam)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    # Show image with boxes if any predictions exist
    if predicted_bboxes is not None: 
        plot_image_with_boxes(vis, predicted_bboxes, mode = 'pred', ax = axs) 
    else: 
        axs.imshow(vis, cmap=plt.cm.bone)

        
def _compute_ap(recall, precision, ap_method = 'area'):
    """Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    
    Args:
        recall (list): Recall values
        precision (list): precision values
        ap_method (str): Method to compute AP

    Returns:
        ap_area (float): Average Precision computed as area under the precision-recall curve.
        ap_11points (float): Average Precision computed by 11-point interpolation method.
    """
    # ------ AP as the area under the precision-recall curve ------
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Find indices where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum the area under the curve
    ap_area = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    # ------ 11-point interpolation AP calculation ------
    ap_11points = 0.0

    # Evaluate precision at 11 recall points: [0, 0.1, ..., 1]
    for thr in np.arange(0, 1.1, 0.1):
        prec_at_thr = precision[recall >= thr]
        max_prec = prec_at_thr.max() if prec_at_thr.size > 0 else 0.0
        ap_11points += max_prec
    ap_11points /= 11  # Average over the 11 points
        
    return ap_area, ap_11points


def compute_overlap(a, b, iou_method = 'iou'):
    """
    Compute the overlap between two sets of bounding boxes.

    Args:
        a: ndarray of shape (N, 4). Array of N bounding boxes, each defined as [x1, y1, x2, y2].
        b: ndarray of shape (K, 4). Array of K bounding boxes, each defined as [x1, y1, x2, y2].
        iou_method: str, optional. Method to compute overlap:
        - 'iou' (default): Intersection over Union
        - 'iobb_detection': Intersection over the detection box (boxes in `a`)
        - 'iobb_annotation': Intersection over the annotation box (boxes in `b`)

    Returns: 
        iou: ndarray of shape (N, K). Overlap values between each pair of boxes from `a` and `b`.
    """

    # Calculate area of each box in b (K boxes) 
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    # Calculate area of each box in a (N boxes)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])

    # Calculate intersection width:
    # For each box in a, find overlap with each box in b
    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(
        np.expand_dims(a[:, 0], 1), b[:, 0]
    )
    # Calculate intersection height:
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(
        np.expand_dims(a[:, 1], 1), b[:, 1]
    )

    # Clamp intersection widths and heights to zero (no negative overlap)
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    # Calculate intersection area
    intersection = iw * ih
    
    # Calculate union area = area_a + area_b - intersection
    ua = np.expand_dims(area_a, axis=1) + area_b - intersection
    ua = np.maximum(ua, np.finfo(float).eps)
    
    if iou_method == 'iou':
        # compute the intersection over union 
        iou = intersection / ua

    elif iou_method == 'iobb_detection': 
        # compute the intersection over the detected bounding box
        iobb1 = intersection / np.expand_dims(area_a, axis=1)
        iou = np.clip(iobb1, 0.0, 1.0)  # Ensure values are in [0, 1]

    elif iou_method == 'iobb_annotation':
        # compute the intersection over the ground truth bounding box
        iobb2 = intersection / area_b
        iou = np.clip(iobb2, 0.0, 1.0)  # Ensure values are in [0, 1]

    return iou
    
    
def evaluate_metrics(annotations, detections, scores, false_positives, true_positives, iou_threshold, iou_method = 'iou'): 
    """
    Evaluate detections against ground truth annotations and update arrays of scores, false positives and true positives based on IoU criteria.

    Args:
        annotations (np.ndarray): Ground truth bounding boxes, shape (K, 4).
        detections (list or np.ndarray): Detected boxes with scores (shape: (N, 5), each as [x1, y1, x2, y2, score].
        scores (np.ndarray): Array of detection confidence scores accumulated so far.
        false_positives (np.ndarray): Array of false positives accumulated so far (1 if FP, 0 otherwise).
        true_positives (np.ndarray): Array of true positives accumulated so far (1 if TP, 0 otherwise).
        iou_threshold (float): IoU threshold to determine if a detection is a true positive.
        iou_method (str, optional): Method for IoU calculation (default is 'iou').

    Returns:
        scores (np.ndarray): Updated array of detection confidence scores.
        iou_scores (list of float): List of IoU scores for each detection.
        false_positives (np.ndarray): Updated array of false positives.
        true_positives (np.ndarray): Updated array of true positives.
    """
    
    detected_annotations = []

    iou_scores = []
    
    for d in detections:
        scores = np.append(scores, d[4])
        
        if annotations.shape[0] == 0:
            false_positives = np.append(false_positives, 1)
            true_positives = np.append(true_positives, 0)
            continue
            
        overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations, iou_method)        
        assigned_annotation = np.argmax(overlaps, axis=1)
        max_overlap = overlaps[0, assigned_annotation]

        iou_scores.append(max_overlap) 
        
        if (max_overlap >= iou_threshold and assigned_annotation not in detected_annotations):
            false_positives = np.append(false_positives, 0)
            true_positives = np.append(true_positives, 1)
            detected_annotations.append(assigned_annotation)
        else:
            false_positives = np.append(false_positives, 1)
            true_positives = np.append(true_positives, 0)

    return scores, iou_scores, false_positives, true_positives


def get_cumlative_attention(cam, bbox):
    """
    Compute the cumulative attention within a specified bounding box on a given heatmap.

    Args:
        cam (np.ndarray): 2D heatmap.
        bbox (list): Bounding box specified as [x_min, y_min, x_max, y_max].

    Returns:
        float: Sum of the attention values within the bounding box.
    """
    return cam[bbox[1]:bbox[3], bbox[0]:bbox[2]].sum()

    
def extract_bounding_boxes_from_heatmap(heatmap, quantile_threshold=0.98, max_bboxes=3, min_area=230, iou_threshold=0.5): 
    """
    Extract bounding boxes from a heatmap by thresholding high-attention regions on a given heatmap. 
    Adapted from BoundingBoxGenerator class in https://github.com/batmanlab/AGXNet/blob/ee99ef199f1f96f7d0c35336935bd117664e733c/utils.py#L11

    Args:
        heatmap (np.ndarray): 2D heatmap.
        quantile_threshold (float): Quantile threshold for heatmap binarization (default: 0.98).
        max_bboxes (int): Maximum number of bounding boxes to return after NMS (default: 3).
        min_area (int): Minimum area (in pixels) for a connected component to be considered (default: 230).
        iou_threshold (float, optional): IoU threshold used during NMS and overlap filtering (default: 0.5).
        
    Returns:
        list: List of bounding boxes with scores, in the format [x_min, y_min, x_max, y_max, score].
    """

    # Threshold heatmap based on quantile and minimum value
    q = np.quantile(heatmap, quantile_threshold)
    mask = (heatmap > q) & (heatmap > 0.5)
    
    # label connected pixels in the binary mask
    label_im, nb_labels = ndimage.label(mask)

    # find the sizes of connected pixels
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # Remove connected components smaller than min_area
    mask_size = sizes < min_area
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0

    # Re-label after removing small components
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)  # sort objects from large to small

    # generate bounding boxes
    bboxes = []
    for l in range(1, len(labels)):
        slice_x, slice_y = ndimage.find_objects(label_im == l)[0]

        # Validate bounding box dimensions
        if (slice_x.start < slice_x.stop) & (slice_y.start < slice_y.stop):

            if (slice_x.stop-slice_x.start) * (slice_y.stop-slice_y.start) < min_area:
                continue
            
            b = [slice_y.start, slice_x.start, slice_y.stop, slice_x.stop]
            score = get_cumlative_attention(heatmap, b) 

            bboxes.append([slice_y.start, slice_x.start, slice_y.stop, slice_x.stop, score])

    # Sort boxes by score descending
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)

    # Convert to tensor for NMS if there are any detections
    if len(bboxes) > 0:
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

        # Apply Non-Maximum Suppression to reduce overlapping boxes
        keep_indices = nms(bboxes_tensor[:, :4], bboxes_tensor[:, 4], iou_threshold)
        keep_indices = keep_indices[:max_bboxes]
        bboxes = bboxes_tensor[keep_indices]
        
        # remove boxes contain within others
        to_keep = []
        for i in range(len(bboxes)):
            keep = True
            for j in range(len(bboxes)):
                if i != j:
                    box1 = bboxes[i, :4]
                    box2 = bboxes[j, :4]

                    # Compute intersection
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])

                    intersection = max(0, x2 - x1) * max(0, y2 - y1)

                    # Compute areas
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                    # Check if intersection equals the area of the smaller box
                    if intersection >= area1*iou_threshold:
                        keep = False
                        break
            if keep:
                to_keep.append(bboxes[i])

        # Convert back to list
        bboxes = torch.stack(to_keep).tolist() if to_keep else []

    return bboxes
        

def Compute_Heatmaps_patches(model: torch.nn.Module,
                     inputs: torch.Tensor,
                     bag_coords, 
                     bag_info, 
                     visualize_img, 
                     seg_mask, 
                     device, 
                     args=None): 

    
    # (1) Compute the output of the model with mixed precision (more efficient)
    with torch.cuda.amp.autocast():
    
        if args.mil_type == 'pyramidal_mil': # multi-scale MIL models 
            if args.type_scale_aggregator in ['concatenation', 'gated-attention']:

                if args.deep_supervision:
                    output, _ = model(inputs) 
                else: 
                    output = model(inputs) 
                            
                bag_prob = torch.sigmoid(output)
                
            elif args.type_scale_aggregator in ['mean_p', 'max_p']:
                side_outputs = model(inputs)
    
                bag_probs = []
                
                for idx, s in enumerate(args.scales): 
                    bag_probs.append(torch.sigmoid(side_outputs[idx]))
    
                if args.type_scale_aggregator == 'mean_p': 
                    bag_prob = sum(bag_probs)/len(args.scales)
                        
                if args.type_scale_aggregator == 'max_p': 
                    bag_prob = torch.stack(bag_probs).max(dim=0)[0]
                    
        else: # single-level patch-based MIL models 
            output = model(inputs)
    
            bag_prob = torch.sigmoid(output)

    img_h = bag_info['img_height']
    img_w = bag_info['img_width']
    
    if args.mil_type == 'embedding' and (args.pooling_type in ['gated-attention', 'attention', 'pma']):  # single-level patch-based MIL models 

        patch_size = bag_info['patch_size']

        # Get instance-level attention scores from the model
        attention_scores = model.get_patch_scores().detach().cpu().squeeze() # Output shape: (batch_size, unm_patches) 
        attention_scores =  (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + torch.finfo(torch.float16).eps)

        # Initialize empty attention map and a count map to accumulate
        attention_map = torch.zeros(img_h, img_w)
        attention_map_counts = torch.zeros(img_h, img_w)

        # For each patch, add the attention score to corresponding spatial region in the attention map
        for patch_idx in range(len(bag_coords)): 
            
            x,y = bag_coords[patch_idx, :] 
            x = x.item(); y = y.item()

            # Compute spatial region of this patch on the heatmap
            x_start = max(0, x)
            x_end = min(img_w, x+patch_size)
            y_start = max(0, y)
            y_end = min(img_h, y+patch_size)
  
            attention_map[y_start:y_end, x_start:x_end] += attention_scores[patch_idx]

            # Keep track of how many patches contribute to each pixel
            attention_map_counts[y_start:y_end, x_start:x_end] += 1

        # Compute average attention by dividing sum by count, avoiding division by zero
        heatmap = torch.where(attention_map_counts == 0, torch.tensor(0.0), torch.div(attention_map, attention_map_counts))

        # Smooth the heatmap with a Gaussian filter for better visualization
        heatmap = torch.from_numpy(gaussian_filter(heatmap, sigma=10*2))

        # Normalize heatmap only within the segmented mask area; set outside mask to 0
        heatmap = torch.where(torch.tensor(seg_mask, dtype=torch.bool), (heatmap - heatmap[seg_mask != 0].min()) / (heatmap[seg_mask != 0].max() - heatmap[seg_mask != 0].min()), torch.tensor(0.0))

        # Extract bounding boxes from heatmap
        predicted_bboxes = extract_bounding_boxes_from_heatmap(heatmap, quantile_threshold=args.quantile_threshold, max_bboxes=args.max_bboxes, min_area = args.min_area, iou_threshold = args.iou_threshold)

        # dictionary with heatmap and predicted bounding boxes
        heatmaps = {
            "heatmap": heatmap, 
            "pred_bboxes": predicted_bboxes
        }
        
    elif args.mil_type == 'pyramidal_mil' and args.pooling_type in ['gated-attention', 'attention', 'pma']:

        # if the model is nested, process inner attention scores
        if args.nested_model:

            # Get inner attention scores per scale and region
            inner_attentions_dict = model.get_inner_scores()
            patch_attentions_dict = model.get_patch_scores()

            # Get scale-level scores if needed for multi-scale aggregation
            if args.type_scale_aggregator in ['concatenation', 'gated-attention']: 
                scale_scores = model.get_scale_scores().detach().cpu()
    
            aggregated_heatmap = torch.zeros(img_h, img_w) # Initialize multi-scale aggregated heatmap
            
            heatmaps = {} # Dictionary to store heatmaps per scale

            # iterate over scales and attention scores
            for idx, (scale, region_dict) in enumerate(inner_attentions_dict.items()): 

                inner_scores_regions = [] # store pixel-level attention scores per patch

                # Loop through each region's nested pixel-level attention scores 
                for region, inner_scores in region_dict.items():
                    inner_scores = inner_scores.detach().cpu().squeeze()
                    
                    bag_coords_scale = bag_coords
                    patch_size = bag_info['patch_size']

                    ratio = patch_size/scale if scale != 'aggregated' else patch_size/args.scales[0]

                    inner_scores = inner_scores.reshape(math.ceil(ratio), math.ceil(ratio))
                    inner_scores_regions.append(inner_scores)

                # Initialize attention and count maps for this scale
                attention_map = torch.zeros(img_h, img_w)
                attention_map_counts = torch.zeros(img_h, img_w)

                # Accumulate patch attention scores spatially
                for patch_idx in range(len(bag_coords_scale)): 
                    
                    x,y = bag_coords_scale[patch_idx, :]
                    x = x.item(); y = y.item()
            
                    x_start = max(0, x)
                    x_end = min(img_w, x+patch_size)
                    y_start = max(0, y)
                    y_end = min(img_h, y+patch_size)
                            
                    patch_map = F.interpolate(inner_scores_regions[patch_idx].unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=True).detach().cpu().squeeze()

                    patch_score = patch_attentions_dict[scale].detach().cpu().squeeze()
                    patch_map *= patch_score[patch_idx]
            
                    patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + torch.finfo(torch.float16).eps)

                    attention_map[y_start:y_end, x_start:x_end] += patch_map
                    
                    attention_map_counts[y_start:y_end, x_start:x_end] += 1

                if args.type_scale_aggregator == 'gated-attention': 
                    scale_score = scale_scores[0, idx]
                
                elif args.type_scale_aggregator == 'concatenation':
                    scale_score = scale_scores.squeeze()[idx]

                # Average attention values by dividing sum by count
                heatmap = torch.where(attention_map_counts == 0, torch.tensor(0.0), torch.div(attention_map, attention_map_counts))

                # Smooth heatmap with Gaussian filter
                heatmap = torch.from_numpy(gaussian_filter(heatmap, sigma=10*2))

                # Normalize heatmap only within the segmented mask area; set outside mask to 0
                heatmap = torch.where(torch.tensor(seg_mask, dtype=torch.bool), (heatmap - heatmap[seg_mask != 0].min()) / (heatmap[seg_mask != 0].max() - heatmap[seg_mask != 0].min()), torch.tensor(0.0))

                # Extract bounding boxes from heatmap
                predicted_bboxes = extract_bounding_boxes_from_heatmap(heatmap, quantile_threshold=args.quantile_threshold, max_bboxes=args.max_bboxes, min_area = args.min_area, iou_threshold = args.iou_threshold)

                # Store heatmap and bounding boxes for current scale in dict
                if args.type_scale_aggregator in ['concatenation', 'gated-attention']: 
    
                    heatmaps[scale] = {
                        "heatmap": heatmap,
                        "pred_bboxes": predicted_bboxes,
                        "scale_score": scale_score
                    }
                        
                    aggregated_heatmap += heatmap * scale_score
    
                elif args.type_scale_aggregator in ['max_p', 'mean_p']:
                    heatmaps[scale] = {
                        "heatmap": heatmap, 
                        "pred_bboxes": predicted_bboxes
                    }
    
                    aggregated_heatmap += heatmap * (1/len(args.scales))
    
        else:  # Not nested model

            # Get instance-level attention scores for all scales from the model
            scale_attentions_dict = model.get_patch_scores()

            # If scale aggregator uses concatenation or gated-attention, also get scale scores
            if args.type_scale_aggregator in ['concatenation', 'gated-attention']: 
                scale_scores = model.get_scale_scores().detach().cpu()

            # Initialize multi-scale aggregated heatmap 
            aggregated_heatmap = torch.zeros(img_h, img_w)

            # Dictionary to store heatmaps and bounding boxes per scale
            heatmaps = {} 

            # Loop over each scale and its corresponding attention scores
            for idx, (scale, attention_scores) in enumerate(scale_attentions_dict.items()): 

                # Get scale-level weight depending on aggregator type
                if args.type_scale_aggregator == 'gated-attention': 
                    scale_score = scale_scores[0, idx]
                    
                elif args.type_scale_aggregator == 'concatenation':
                    scale_score = scale_scores.squeeze()[idx]

                attention_scores = attention_scores.detach().cpu().squeeze()

                # Handle coordinate and patch size depending on multi-scale model type
                if args.multi_scale_model == 'msp': 
                    bag_coords_scale = bag_coords[scale] if scale != 'aggregated' else bag_coords[args.scales[0]]
                    patch_size = bag_info[scale]['patch_size'] if scale != 'aggregated' else args.scales[0]
    
                elif args.multi_scale_model in ['fpn', 'backbone_pyramid']:         
                    bag_coords_scale = bag_coords
                    patch_size = bag_info['patch_size']

                    # Calculate ratio for reshaping pixel-level attention scores spatially
                    ratio = patch_size/scale if scale != 'aggregated' else patch_size/args.scales[0]
                    attention_scores = attention_scores.reshape(len(bag_coords_scale), math.ceil(ratio), math.ceil(ratio)) 

                # Initialize empty tensors for accumulating attention values and counts
                attention_map = torch.zeros(img_h, img_w)
                attention_map_counts = torch.zeros(img_h, img_w)
    
                flag = 0 

                # Loop over each patch coordinate for the current scale
                for patch_idx in range(len(bag_coords_scale)): 

                    # Get x,y coordinates of the patch (top-left corner)
                    x,y = bag_coords_scale[patch_idx, :]
                    x = x.item(); y = y.item()
        
                    x_start = max(0, x)
                    x_end = min(img_w, x+patch_size)
                    y_start = max(0, y)
                    y_end = min(img_h, y+patch_size)
                        
                    if args.multi_scale_model in ['fpn', 'backbone_pyramid']:
                        
                        # Upsample the spatial attention patch map to full patch size
                        patch_map = F.interpolate(attention_scores[patch_idx].unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=True).detach().cpu().squeeze()

                        # Normalize patch_map to [0,1]
                        patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + torch.finfo(torch.float16).eps)

                        # Add normalized patch attention to the aggregated attention map for this scale
                        attention_map[y_start:y_end, x_start:x_end] += patch_map
                        
                    elif args.multi_scale_model == 'msp':
                        # Directly add scalar patch-level attention score for this patch to the attention map region
                        attention_map[y_start:y_end, x_start:x_end] += attention_scores[patch_idx]
                        
                    attention_map_counts[y_start:y_end, x_start:x_end] += 1

                # Compute average attention per pixel
                heatmap = torch.where(attention_map_counts == 0, torch.tensor(0.0), torch.div(attention_map, attention_map_counts))

                # Apply Gaussian smoothing
                heatmap = torch.from_numpy(gaussian_filter(heatmap, sigma=10))

                # Normalize heatmap values only inside the segmentation mask, zero outside
                heatmap = torch.where(torch.tensor(seg_mask, dtype=torch.bool), (heatmap - heatmap[seg_mask != 0].min()) / (heatmap[seg_mask != 0].max() - heatmap[seg_mask != 0].min()), torch.tensor(0.0))

                # Extract bounding boxes from heatmap
                predicted_bboxes = extract_bounding_boxes_from_heatmap(heatmap, quantile_threshold=args.quantile_threshold, max_bboxes=args.max_bboxes, min_area = args.min_area, iou_threshold = args.iou_threshold)

                # Store heatmap and bounding boxes for each scale
                if args.type_scale_aggregator in ['concatenation', 'gated-attention']: 
    
                    heatmaps[scale] = {
                        "heatmap": heatmap,
                        "pred_bboxes": predicted_bboxes,
                        "scale_score": scale_score
                    }
                        
                    aggregated_heatmap += heatmap * scale_score
    
                elif args.type_scale_aggregator in ['max_p', 'mean_p']:
                    heatmaps[scale] = {
                        "heatmap": heatmap, 
                        "pred_bboxes": predicted_bboxes
                    }

                    aggregated_heatmap += heatmap * (1/len(args.scales))

        # If aggregated heatmap is not already included in heatmaps dict
        if 'aggregated' not in heatmaps:

            # Normalize aggregated heatmap to [0,1]
            aggregated_heatmap = (aggregated_heatmap - aggregated_heatmap.min()) / (aggregated_heatmap.max() - aggregated_heatmap.min())  

            # Extract bounding boxes from aggregated heatmap
            predicted_bboxes = extract_bounding_boxes_from_heatmap(aggregated_heatmap, quantile_threshold=args.quantile_threshold, max_bboxes=args.max_bboxes, min_area = args.min_area, iou_threshold = args.iou_threshold)

            # Add aggregated heatmap and bboxes to heatmaps dictionary
            heatmaps["aggregated"] = {
                    "heatmap": aggregated_heatmap,
                    "pred_bboxes": predicted_bboxes
                }
        
    return bag_prob, heatmaps

    
def Visualize_ROI_Eval(model, 
                       dataloader, 
                       device, 
                       df,
                       output_dir, 
                       fig_name, 
                       args):

    # Define a reverse normalization transform to convert normalized images back to original range
    reverse_transform = transforms.Compose([
        transforms.Normalize((-args.mean / args.std, -args.mean / args.std, -args.mean / args.std), (1.0 / args.std, 1.0 / args.std, 1.0 / args.std))
    ])

    # Setup the figure and axes layout depending on model type and scale aggregator
    if args.mil_type == 'pyramidal_mil':
        fig, axs = plt.subplots(
            args.visualize_num_images, 
            len(args.scales) + 2, 
            figsize=(30, 20), 
            gridspec_kw={'width_ratios': [1] * (len(args.scales)+1) + [1.1]}
        )
        
    else: # single-level patch-based mil baselines 
        fig, axs = plt.subplots(args.visualize_num_images, 2, figsize=(20, 10))
    
    plot_idx = 0 
    
    for idx, data in enumerate(tqdm(dataloader)):

        bag_info = data['bag_info'][0]
        img_path = bag_info['img_dir']
        
        if img_path in df['img_path'].values: 

            # Send data to device
            if isinstance(data['x'], list): 
                inputs = [tensor.to(device, non_blocking=True) for tensor in data['x']]
            elif isinstance(data['x'], dict):
                inputs = {scale: tensor.to(device, non_blocking=True) for scale, tensor in data['x'].items()}
            else: 
                inputs = data['x'].to(device, non_blocking=True)
            
            target = data['y'].to(device, non_blocking=True).float()
            boxes = df[df['img_path'] == img_path]['boxes'].values[0] # Retrieve gt bounding boxes for this image from dataframe

            bag_coords = data['coords'] # Load patch coordinates associated with this bag

            # load input image in RGB mode
            img = Image.open(bag_info['img_dir']).convert('RGB')
            image_rgb, padding = pad_image(transforms.ToTensor()(img), args.patch_size)
            padding_left, padding_right, padding_top, padding_bottom = padding

            # Segment image to create segmentation mask, then pad it the same way as image
            seg_mask = Segment(transforms.ToTensor()(img))
            seg_mask, _ = pad_image(seg_mask, args.patch_size) 

            # Retrieve heatmaps and bag-level probability
            bag_prob, heatmaps = Compute_Heatmaps_patches(model, inputs, bag_coords, bag_info, False, seg_mask, device, args)

            image_rgb /= image_rgb.max()
                    
            if args.mil_type == 'pyramidal_mil': 

                # Plot original image with bounding boxes
                plot_image_with_boxes(image_rgb.permute(1,2,0), boxes, ax = axs[plot_idx][0])
                axs[plot_idx][0].set_title(Get_Predicted_Class(target, bag_prob, label_type = args.label), fontsize=16)
                axs[plot_idx][0].axis('off')

                # Loop over each scale and its heatmap to plot corresponding heatmaps and bounding boxes
                for idx, (scale, data) in enumerate(heatmaps.items()): 

                    # Access IoU score for current scale from dataframe
                    iou_score = df[df['img_path'] == img_path][f'iou_score_{scale}'].values[0]
                    
                    if scale == "aggregated": 
                        # Show multi-scale aggregated heatmap with predicted bounding boxes and IoU score
                        attention_vis = ShowVis(data["heatmap"], image_rgb.permute(1,2,0), data["pred_bboxes"], axs[plot_idx][idx+1], args)
                        axs[plot_idx][idx+1].set_title(f"Aggregated Heatmap (IoU: {iou_score*100:.2f})", fontsize=16)
                        axs[plot_idx][idx+1].axis('off')
                        
                    else: 
                        # Show scale-specific heatmap with predicted bounding boxes and IoU score
                        attention_vis = ShowVis(data["heatmap"], image_rgb.permute(1,2,0), data["pred_bboxes"], axs[plot_idx][idx+1], args)
                        
                        plot_text = f"Scale {scale}x - Heatmap (IoU: {iou_score*100:.2f})"
                        axs[plot_idx][idx+1].set_title(plot_text, fontsize=16)
                        axs[plot_idx][idx+1].axis('off')

                # Add colorbar on the last heatmap subplot
                cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), 
                                    ax=axs[plot_idx][len(args.scales)+1], 
                                    orientation='vertical', 
                                    fraction=0.046, 
                                    pad=0.04)

            else: # single-scale patch-based mil models 
                
                iou_score = df[df['img_path'] == img_path][f'iou_score'].values[0]
                
                # Plot input image with bounding boxes
                plot_image_with_boxes(image_rgb.permute(1,2,0), boxes, ax = axs[plot_idx][0])
                axs[plot_idx][0].set_title(Get_Predicted_Class(target, bag_prob, label_type = args.label), fontsize=16)
                axs[plot_idx][0].axis('off')

                # Plot heatmap with predicted bounding boxes
                ShowVis(heatmaps['heatmap'], image_rgb.permute(1,2,0), heatmaps['pred_bboxes'], axs[plot_idx][1], args)  
                axs[plot_idx][1].set_title(f"Heatmap (IoU: {iou_score*100:.2f})", fontsize=16)
                axs[plot_idx][1].axis('off')

                # Add colorbar on the last heatmap subplot
                cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=axs[args.visualize_num_images-1][1], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=10)
        
                plt.tight_layout()

            # Clean up variables to save memory
            del img, image_rgb, seg_mask, heatmaps; clear_memory()

            # Increment row index
            plot_idx += 1

        # Stop after visualizing specified number of images
        if plot_idx >= args.visualize_num_images:
            break 

    plt.savefig(os.path.join(output_dir, fig_name), bbox_inches='tight')
    plt.close(fig)

    del fig; clear_memory()

def roi_categorization(boxes, args): 

    category_boxes = []

    # Iterate over each bounding box 
    for box in boxes:
        xmin, ymin, xmax, ymax = box

        # Calculate the area of the bounding box
        area = (xmax - xmin) * (ymax - ymin)
        
        # Categorize the box based on its area
        if area < 128**2 and args.roi_eval_scheme == 'small_roi':
            category_boxes.append([xmin, ymin, xmax, ymax])
        elif 128**2 <= area < 256**2 and args.roi_eval_scheme == 'medium_roi':
            category_boxes.append([xmin, ymin, xmax, ymax])
        elif area >= 256**2 and args.roi_eval_scheme == 'large_roi':
            category_boxes.append([xmin, ymin, xmax, ymax])

    if category_boxes: 
        category_boxes = np.array(category_boxes)
    else: 
        category_boxes = None 
        
    return category_boxes

        
def run_roi_eval(directory, args, device):

    if args.feature_extraction == 'online': 
        if 'efficientnetv2' in args.arch:
            args.model_base_name = 'efficientv2_s'
        elif 'efficientnet_b5_ns' in args.arch:
            args.model_base_name = 'efficientnetb5'
        else:
            args.model_base_name = args.arch
        
    args.n_class = 1

    # Task specificities
    if args.label.lower() == 'mass':
        class0 = 'not_mass'
        class1 = 'mass'
    elif args.label.lower() == 'suspicious_calcification':
        class0 = 'not_calcification'
        class1 = 'calcification'   

    label_dict = {class0: 0, class1: 1}

    # Prepare output directory for ROI visualizations
    roi_dir = os.path.join(directory, 'roi_visualization_new', args.dataset, args.roi_eval_set, args.roi_eval_scheme, f'{args.iou_method}_threshold_{args.iou_threshold}')
    os.makedirs(roi_dir, exist_ok=True)

    ############################ Data Setup ############################
    args.data_dir = Path(args.data_dir)
    
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    
    if args.dataset == 'ViNDr':
        if args.roi_eval_set == 'val': 
            dev_df = args.df[args.df['split'] == "training"].reset_index(drop=True)
            _, test_df = stratified_train_val_split(dev_df, 0.2, args = args)
    
        elif args.roi_eval_set == 'test': 
            test_df = args.df[args.df['split'] == "test"].reset_index(drop=True)

    dataloader = MIL_dataloader(test_df ,'test', args)

    # Load model and freeze parameters
    model = build_model(args)
    for param in model.parameters():
        param.requires_grad = False
    model.is_training = False 
    model.to(device)

    # Load checkpoint and set model to eval mode
    checkpoint = torch.load(os.path.join(directory, 'best_model.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    # DETECTION PERFORMANCE 
    if args.mil_type == 'pyramidal_mil': 
        false_positives = {}
        true_positives = {}
        scores = {}

        for s in args.scales: 
            false_positives[s] = np.zeros((0,))
            true_positives[s] = np.zeros((0,))
            scores[s] = np.zeros((0,))

        false_positives['aggregated'] = np.zeros((0,))
        true_positives['aggregated'] = np.zeros((0,))
        scores['aggregated'] = np.zeros((0,))
            
    else: # single-scale patch-based mil models 
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
    
    num_annotations = 0.0

    # DataFrame to store IoU evaluation results per image
    iou_df = pd.DataFrame(columns=["img_path", "boxes"] + [f"iou_score_{scale}" for scale in args.scales])

    total_num_imgs = 0 

    # Define reverse transform to visualize heatmaps on original image
    reverse_transform = transforms.Compose([
        transforms.Normalize((-args.mean / args.std, -args.mean / args.std, -args.mean / args.std), (1.0 / args.std, 1.0 / args.std, 1.0 / args.std))
    ])
    
    # Iterate over the dataloader
    for idx, data in enumerate(tqdm(dataloader)):

        # Send data to device
        if isinstance(data['x'], list): 
            inputs = [tensor.to(device, non_blocking=True) for tensor in data['x']]
        elif isinstance(data['x'], dict):
            inputs = {scale: tensor.to(device, non_blocking=True) for scale, tensor in data['x'].items()}
        else: 
            inputs = data['x'].to(device, non_blocking=True)
            
        target = data['y'].to(device, non_blocking=True).float()
        boxes = data['boxes'] 

        if not args.roi_eval_scheme == 'all_roi': 
            boxes = roi_categorization(boxes, args)

            if boxes is None: 
                continue 

        bag_coords = data['coords']
        bag_info = data['bag_info'][0]

        # Load and pad image
        img = Image.open(bag_info['img_dir']).convert('RGB')
        image_rgb, padding = pad_image(transforms.ToTensor()(img), args.patch_size)
        seg_mask = Segment(transforms.ToTensor()(img))
        seg_mask, _ = pad_image(seg_mask, args.patch_size)        

        # Compute heatmaps and predicted boxes
        bag_prob, heatmaps = Compute_Heatmaps_patches(model, inputs, bag_coords, bag_info, total_num_imgs < args.visualize_num_images, seg_mask, device, args)
        
        num_annotations += boxes.shape[0]

        iou_df_new = {
            "img_path": bag_info['img_dir'],
            "boxes": boxes,
        }

        ############################ IoU Evaluation ############################
        if args.mil_type == 'pyramidal_mil': 
            
            for scale, heatmap in heatmaps.items(): 
                
                scores[scale], iou_scores, false_positives[scale], true_positives[scale] = evaluate_metrics(boxes, heatmap['pred_bboxes'], scores[scale], false_positives[scale], true_positives[scale], args.iou_threshold, args.iou_method)
                    
                iou_df_new[f"iou_score_{scale}"] = np.max(iou_scores)
        
        else: # single-scale patch-based mil models 
            scores, iou_scores, false_positives, true_positives = evaluate_metrics(boxes, heatmaps['pred_bboxes'], scores, false_positives, true_positives, args.iou_threshold, args.iou_method)
                    
            iou_df_new[f"iou_score"] = np.max(iou_scores)

        
        # Add the entry to the DataFrame
        iou_df = pd.concat([iou_df, pd.DataFrame([iou_df_new])], ignore_index=True)

        
        ############################ Visualization ############################
        if total_num_imgs < args.visualize_num_images: 
            
            image_rgb /= image_rgb.max()
            
            if args.mil_type == 'pyramidal_mil': 
                
                fig, axs = plt.subplots(
                    1, 
                    len(args.scales) + 2, 
                    figsize=(30, 20), 
                    gridspec_kw={'width_ratios': [1] * (len(args.scales)+1) + [1.1]} 
                )

                # Plot the input image
                plot_image_with_boxes(image_rgb.permute(1,2,0), boxes, ax = axs[0])
                axs[0].set_title(Get_Predicted_Class(target, bag_prob, label_type = args.label), fontsize=16)
                axs[0].axis('off')
                
                for idx, (scale, data) in enumerate(heatmaps.items()):

                    iou_score = iou_df_new[f'iou_score_{scale}']
                    
                    if scale == 'aggregated': 
                        attention_vis = ShowVis(data['heatmap'], image_rgb.permute(1,2,0), data['pred_bboxes'], axs[idx+1], args)
                        axs[idx+1].set_title(f"Heatmap Map (IoU: {iou_score*100:.2f})", fontsize=16)
                        axs[idx+1].axis('off')
                        
                    else: 
                        
                        attention_vis = ShowVis(data['heatmap'], image_rgb.permute(1,2,0), data['pred_bboxes'], axs[idx+1], args)
                        
                        plot_text = f"Scale {scale}x - Heatmap (IoU: {iou_score*100:.2f})"
                        axs[idx+1].set_title(plot_text, fontsize=16)
                        axs[idx+1].axis('off')

                cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), 
                                    ax=axs[len(args.scales)+1], 
                                    orientation='vertical', 
                                    fraction=0.046, 
                                    pad=0.04)
                
                cbar.ax.tick_params(labelsize=10)
        
            else: # single-scale patch-based mil models 

                fig, axs = plt.subplots(1,2, figsize=(10, 5))

                # (1) Plot the input image
                plot_image_with_boxes(image_rgb.permute(1,2,0), boxes, ax = axs[0])
                axs[0].set_title(Get_Predicted_Class(target, bag_prob, label_type = args.label), fontsize=16)
                axs[0].axis('off')
                
                ShowVis(heatmaps['heatmap'], image_rgb.permute(1,2,0), heatmaps['pred_bboxes'], axs[1], args)  
                axs[1].set_title(f"Heatmap (IoU: {np.max(iou_scores)*100:.4f})", fontsize=16)
                axs[1].axis('off')
    
                # Add colorbar
                cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=axs[1], fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=10)
    
            plt.tight_layout()
            #plt.show()

            roi_dir_all = os.path.join(roi_dir, 'all_heatmaps')
            os.makedirs(roi_dir_all, exist_ok=True)

            fig_name = os.path.basename(bag_info['img_dir'])
            
            plt.savefig(os.path.join(roi_dir_all, fig_name), bbox_inches='tight')

            if total_num_imgs < args.visualize_num_images:
                plt.show()

            plt.close(fig)

            total_num_imgs += 1 

                     
        del img, image_rgb, seg_mask, heatmaps; clear_memory()

    ############################ Compute mAP ############################
    print('Total number of annotations:', num_annotations) 

    if args.mil_type == 'pyramidal_mil':
        results = {scale: [] for scale in args.scales}
        results['aggregated'] = [] 
        
        # sort by score
        for s in scores: 
            indices = np.argsort(-scores[s])
            false_positives[s] = false_positives[s][indices]
            true_positives[s] = true_positives[s][indices]
        
            # compute false positives and true positives
            false_positives[s] = np.cumsum(false_positives[s])
            true_positives[s] = np.cumsum(true_positives[s])
        
            # compute recall and precision
            recall = true_positives[s] / num_annotations
            precision = true_positives[s] / np.maximum(
                true_positives[s] + false_positives[s], np.finfo(np.float16).eps
            )
        
            # compute average precision
            average_precision_area, average_precision_11points = _compute_ap(recall, precision, args.ap_method)
    
            # Collect results for this scale
            results[s].append(average_precision_area)

    else: 
        results = []
        
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]
        
        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)
        
        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(
            true_positives + false_positives, np.finfo(np.float16).eps
        )
        
        # compute average precision
        average_precision_area, average_precision_11points = _compute_ap(recall, precision, args.ap_method)
        
        # Collect results for this scale
        results.append(average_precision_area)
        
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    print(results_df)
    
    results_df.to_csv(os.path.join(roi_dir, 'roi_eval_results.csv'), index = False)

    del indices, false_positives, true_positives, recall, precision, average_precision_area, average_precision_11points; clear_memory()

    ############################ Optional Visualizations ############################
    if args.visualize_num_images: 
        comparison_cases = iou_df.sample(n=args.visualize_num_images, random_state=3)

        Visualize_ROI_Eval(model, 
                           dataloader, 
                           device, 
                           df = comparison_cases,
                           output_dir = roi_dir, 
                           fig_name = 'comparison_cases_heatmaps.png', 
                           args=args)

        if args.mil_type == 'pyramidal_mil':
            # Top, worst and comparison cases
            best_cases = iou_df.sort_values(by="iou_score_aggregated", ascending=False).head(args.visualize_num_images)
            worst_cases = iou_df.sort_values(by="iou_score_aggregated", ascending=True).head(args.visualize_num_images)
        
        else: # single-scale patch-based mil models 
            best_cases = iou_df.sort_values(by="iou_score", ascending=False).head(args.visualize_num_images)
            worst_cases = iou_df.sort_values(by="iou_score", ascending=True).head(args.visualize_num_images)
            
        Visualize_ROI_Eval(model, 
                           dataloader, 
                           device, 
                           df = best_cases,
                           output_dir = roi_dir, 
                           fig_name = 'best_cases_heatmaps.png', 
                           args=args)
    
        Visualize_ROI_Eval(model, 
                           dataloader, 
                           device, 
                           df = worst_cases,
                           output_dir = roi_dir, 
                           fig_name = 'worst_cases_heatmaps.png', 
                           args=args)

    return results_df

    
def ROI_Eval(args, device):
    """
    Perform Region of Interest (ROI) evaluation for multiple model runs.

    Args:
        args: argparse.Namespace containing configuration and hyperparameters.
        device: The device (CPU or GPU) on which to run the model.
    """
    
    all_results = []  # Store results from all runs

    # Loop over the number of model runs specified
    for run_idx in range(args.n_runs):

        # Set random seed for reproducibility
        seed_all(args.seed)
        
        print(f'\nRunning roi eval for model run nº{run_idx + args.start_run}....')

        # Construct the path to the saved model run
        run_path = os.path.join(args.resume, f'run_{args.start_run + run_idx}')

        # Perform the ROI evaluation and return the results as a DataFrame
        run_results_df = run_roi_eval(run_path, args, device)  
            
        # Add column to track the run
        run_results_df["runs"] = args.start_run + run_idx
            
        all_results.append(run_results_df)

    # If multiple runs were performed, aggregate and summarize results
    if args.n_runs > 1: 

        # Combine all runs into a single DataFrame
        combined_df = pd.concat(all_results, ignore_index=True)

        # Calculate mean and std for specific columns
        mean_std = combined_df.drop('runs', axis=1).agg(['mean', 'std']).reset_index(drop=True)
        mean_std['runs'] = ['mean', 'std']

        # Append mean and std to the original DataFrame
        combined_df = pd.concat([combined_df, mean_std]).reset_index(drop=True)

        print(combined_df)

        # Save the summarized evaluation results to CSV
        output_path = os.path.join(args.resume, f'{args.roi_eval_set}_{args.roi_eval_scheme}_eval_summary.csv')
        combined_df.to_csv(output_path, index=False)

    

    

