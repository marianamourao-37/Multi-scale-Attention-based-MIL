# Multi-scale Attention-based Multiple Instance Learning (MIL)

This repository contains the **official implementation** of the paper:
**"Multi-scale Attention-based Multiple Instance Learning for Breast Cancer Diagnosis"**  
Accepted at the **MICCAI 2025** conference.

## Repository Structure

```plaintext
Multi-scale-Attention-based-MIL/
│── Datasets/                  
│   ├── dataset_concepts.py                 # Dataset classes for MIL classification & lesion detection   
│   └── dataset_utils.py                    # Data preprocessing & loading    
│── Feature_Extractors/                   
│   ├── __init__.py                         # Load feature extractor  
│   ├── FPN.py                              # Feature Pyramid Network
│   └── mammoclip/                          
│       ├── __init__.py                     # Load pre-trained MammoCLIP image encoder
│       ├── efficient_net_custom_utils.py   # Helper functions for EfficientNet 
│       └── efficientnet_custom.py          # Modified EfficientNet implementation 
│── MIL/                                    
│   ├── __init__.py                         # Build & configure MIL model
│   ├── MIL_models.py                       # MIL model architectures
│   ├── AttentionModels.py                  # Attention-based MIL modules   
│   ├── MIL_experiment.py                   # MIL classification training & evaluation 
│   ├── inference_MIL_classifier.py         # MIL classification inference   
│   └── roi_eval.py                         # Lesion detection MIL evaluation 
│── utils/                                  
│   ├── data_split_utils.py                 # Data splitting utilities    
│   ├── generic_utils.py                    # General-purpose utilities 
│   ├── plot_utils.py                       # Plotting & visualization 
│   ├── training_setup_utils.py             # Training setup & configuration
│   └── metrics.py                          # Evaluation metrics 
│── vindrmammo_grouped_df.csv               # Dataset metadata (e.g. file names & labels)  
│── main.py                                 # Main script to train & evaluate implemented models 
│── offline_feature_extraction.py           # Script for offline feature extraction  
````

## ⏳ Coming Soon (under construction) 
- [ ] Scripts for training and evaluating **MIL classification tasks**
- [ ] Scripts for post-hoc evaluation of **lesion detection tasks**
- [ ] Jupyter Notebooks to reproduce **MICCAI 2025 results**
