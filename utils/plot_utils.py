import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics 

def plot_loss_and_acc_curves(results_train, results_val, metric_name, path_results_fold, model_ema=False):
    """Plots training curves of a results dictionary.

    """
    train_loss = results_train['loss']
    val_loss = results_val['loss']

    train_bacc = results_train[metric_name]
    val_bacc = results_val[metric_name]

    epochs = np.arange(1, len(results_val['loss']) + 1)

    index_loss = np.argmin(val_loss) # this is the epoch with the lowest validation loss
    val_lowest = val_loss[index_loss]
    
    index_bacc = np.argmax(val_bacc)
    bacc_highest = val_bacc[index_bacc]
    
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss + 1)
    vc_label='best epoch= '+ str(index_bacc + 1)

    fig, axs=plt.subplots(nrows=1, ncols=2, figsize=(20,8))

    axs[0].plot(epochs, train_loss, 'r', label="Train Loss")
    axs[0].plot(epochs, val_loss, 'g', label="Val. Loss")
    axs[0].scatter(index_loss+1, val_lowest, s=150, c= 'blue', label=sc_label)
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epochs, train_bacc, 'r', label=f"Train {metric_name}")
    axs[1].plot(epochs, val_bacc, 'g', label=f"Val. {metric_name}")
    axs[1].scatter(index_bacc+1, bacc_highest, s=150, c= 'blue', label=vc_label)
    axs[1].set_title(f'Training and Validation {metric_name}')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel(f'{metric_name}')
    
    axs[1].legend()
    
    plt.tight_layout()

    model = ''
    if model_ema:
        model = 'model_ema'
    # Save the figure
    plt.savefig(path_results_fold / 'loss_curves.png')
    plt.clf()

def plot_lrs_scheduler(lrs_experimental, output_dir):

    plt.figure()
    plt.plot(np.arange(1, len(lrs_experimental)+1), lrs_experimental, 'r')
    plt.title('Experimental LR Scheduler')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / 'LRscheduler.png')
    plt.clf()

def plot_confusion_matrix(confusion_matrix, class_names, checkpoint_name, output_dir, model_ema=False):

    # Create a DataFrame for the confusion matrix
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)

    # Set the figure size to ensure the plot is not stretched
    plt.figure(figsize=(10, 8))
    
    # Create the heatmap
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap='Blues')

    # Adjust the tick labels and their appearance
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right',fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right',fontsize=15)
    
    # Set labels for axes
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout() # Adjust layout to prevent clipping of tick labels

    model = ''
    if model_ema:
        model = '_model_ema'
        
    plt.savefig(output_dir / f'{checkpoint_name}-confusion_matrix.png', bbox_inches='tight')
    plt.clf()

def ROC_curves(test_targs, test_probs, checkpoint_name, output_dir, model_ema=False):

    model = ''
    if model_ema:
        model = '_model_ema'
        
    # calculate roc curve
    fpr, tpr, roc_thresholds = metrics.roc_curve(test_targs, test_probs, drop_intermediate=False) 

    # calculate ROC-AUC
    roc_auc = metrics.auc(fpr, tpr)
    
    # calculate precision-recall curve
    precision, recall, pr_thresholds = metrics.precision_recall_curve(test_targs, test_probs, drop_intermediate=False) 

    # calculate precision-recall AUC
    pr_auc = metrics.auc(recall, precision)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([-0.1, 1.0])
    plt.ylim([-0.1, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / f'{checkpoint_name}-roc_curve.png', bbox_inches='tight')
    plt.clf()

    # plot the precision-recall curves
    plt.figure()
    no_skill = len(test_targs[test_targs==1]) / len(test_targs)
    plt.plot([0, 1], [no_skill, no_skill], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.3f)' % pr_auc)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    # show the legend
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_dir / f'{checkpoint_name}-pr_curve.png', bbox_inches='tight')
    plt.clf()
