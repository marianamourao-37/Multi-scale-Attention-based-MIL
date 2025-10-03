# Multi-scale Attention-based Multiple Instance Learning (MIL)

This repository contains the **official implementation** of the paper:

📄 [**"Multi-scale Attention-based Multiple Instance Learning for Breast Cancer Diagnosis"**](https://link.springer.com/chapter/10.1007/978-3-032-05182-0_36)

Accepted at the **MICCAI 2025** conference for oral presentation and poster session. 

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

# Data Download

As mentioned in the paper, this work uses the preprocessed images provided by Ghosh et al. in their Mammo-CLIP work. 
- [Link to input images](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)

Regarding metadata information for the downstream tasks (image classification and lesion detection), please use the **vindrmammo_grouped_df.csv** provided in this repository. 

# Feature Extraction 

Following prior deep MIL models that handle large-size bags, the implemented framework uses the pretrained EfficientNet-B2 image encoder from the Mammo-CLIP work as the backbone for feature extraction. 
- [Link for the pretrained EfficientNet-B2](https://huggingface.co/shawn24/Mammo-CLIP/blob/main/Pre-trained-checkpoints/b2-model-best-epoch-10.tar)

After successfully downloading the image encoder checkpoint, you need to set the --clip_chk_pt_path argument to the correct path. 

The implemented framework is compatible with both online and offline feature extraction. To perform offline feature extraction, run the following code:

```bash
python offline_feature_extraction.py \
  --clip_chk_pt_path "foundational_models/Mammo-CLIP-main/b2-model-best-epoch-10.tar" \ # Path to Mammo-CLIP's image encoder checkpoint
  --dataset 'ViNDr' \
  --arch 'upmc_breast_clip_det_b2_period_n_lp' \
  --csv-file 'vindrmammo_grouped_df.csv' \
  --feat_dir 'PreProcessedData/Vindir-mammoclip/extracted_features' \
  --patching \ # Wether to perform patching on full-resolution images. If false, it will consider previously extracted patches that were saved in a directory
  --patch_size 512 \ 
  --overlap 0.0 \
  --multi_scale_model 'fpn'
```

# Checkpoints

We provide pre-training checkpoints for our best-performing models.  

|Description         | Checkpoints |
|--------------------|-------------|
| Best model for calcifications | [FPN-SetTrans](https://drive.google.com/file/d/1pcr5wa8cI7R8L-7MfkXBEBB2IE02NmMI/view?usp=sharing) |
| Best model for masses | [FPN-AbMIL](https://drive.google.com/file/d/1ptgub09TjB2oCpm2ij2OyaVDKT_5y8D0/view?usp=sharing) |

## ⏳ Coming Soon (under construction) 
- [ ] Scripts for training and evaluating **MIL classification tasks**
- [ ] Scripts for post-hoc evaluation of **lesion detection tasks**
- [ ] Jupyter Notebooks to reproduce **MICCAI 2025 results**

## Reference 

If you find our work useful in your research or if you use parts of this code, please use the following BibTeX entry.

```plaintext
@InProceedings{MouMar_Multiscale_MICCAI2025,
        author = { Mourão, Mariana AND Nascimento, Jacinto C. AND Santiago, Carlos AND Silveira, Margarida},
        title = { { Multi-scale Attention-based Multiple Instance Learning for Breast Cancer Diagnosis } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15974},
        month = {September},
        page = {364 -- 374}
}
````
