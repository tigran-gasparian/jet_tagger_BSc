import os, shutil, json
import numpy as np
import tensorflow as tf
#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
#from schema import Schema, And, Use, Optional
import shap
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import plot.style as style
#import collections
#import keras
#from keras.layers import BatchNormalization,  Activation, AveragePooling1D, Dense, Conv1D, Flatten
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable


#functions for plotting
style.set_style()

def ROC(y_pred, y_test, class_labels, plot_dir, ROC_dict, suffix=None):
    """
        plots ROC curves for all different classes and saves in output directory 
    """
    # Create a colormap for unique colors
    colormap = matplotlib.colormaps['Set1'].resampled(len(class_labels))  # Use 'tab10' with enough colors

    # Create a plot for ROC curves
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax, fontsize=style.CMSHEADER_SIZE)
    for i, class_label in enumerate(class_labels):

        # Get true labels and predicted probabilities for the current class
        y_true = y_test[:, i]  # Extract the one-hot column for the current class
        y_score = y_pred[:, i] # Predicted probabilities for the current class

        # Compute FPR, TPR, and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ROC_dict[class_label] = roc_auc
        # Plot the ROC curve for the current class
        ax.plot(tpr, fpr, label=f'{style.CLASS_LABEL_STYLE[class_label]} (AUC = {roc_auc:.2f})',
                 color=colormap(i), linewidth=style.LINEWIDTH)

    # Plot formatting
    ax.grid(True)
    ax.set_ylabel('Mistag Rate')
    ax.set_xlabel('Signal Efficiency')

    auc_list = [value for key,value in ROC_dict.items()]
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.argsort(auc_list)
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc='upper left',ncol=2,fontsize=style.SMALL_SIZE-3)

    ax.set_yscale('log')
    ax.set_ylim([1e-3, 1.1])

    # Save the plot
    save_path = os.path.join(plot_dir, "basic_ROC" + (f"_{suffix}" if suffix is not None else ""))
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

    return ROC_dict
def confusion(y_pred, y_test, class_labels, plot_dir):
    """
    plots confusion matrix and saves in output directory
    """
    
    cm = confusion_matrix(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1),
        normalize="true",
        )
    cm = np.round(cm, 3)
    class_labels = {v: k for k, v in class_labels.items()}
    labels = [style.CLASS_LABEL_STYLE[class_labels[i]] for i in range(y_test.shape[1])]

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, fontsize=style.CMSHEADER_SIZE)
    matrix_display = ConfusionMatrixDisplay(cm, display_labels=labels)

    matrix_display.plot(ax=ax)
    matrix_display.im_.set_clim(0, 1)

    # Remove default the colorbar
    matrix_display.im_.colorbar.remove()

    # Adjust colorbar height to match the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(matrix_display.im_, cax=cax)

    # Save the plot
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix.png"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix.pdf"), bbox_inches='tight')

def shapPlot(shap_values, feature_names, class_names):
    """
    Calculates shap values for all classes
    """
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
    num_features = (shap_values[0].shape[1])
    feature_inds = feature_order
    y_pos = np.arange(len(feature_inds))
    left_pos = np.zeros(len(feature_inds))
    axis_color="#333333"
    class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
    colormap = matplotlib.colormaps['Set1'].resampled(len(class_names))  # Use 'tab10' with enough colors

    for i, ind in enumerate(class_inds):
        global_shap_values = np.abs(shap_values[ind]).mean(0)
        label = style.CLASS_LABEL_STYLE[class_names[ind]]
        ax.barh(y_pos, global_shap_values[feature_inds], 0.7, left=left_pos, align='center',label=label,color=colormap(class_inds[i]))
        left_pos += global_shap_values[feature_inds]

    #ax.set_yticklabels([style.INPUT_FEATURE_STYLE[feature_names[i]] for i in feature_inds])
    ax.legend(loc='lower right',fontsize=30)

    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.set_yticks(range(len(feature_order)), [style.INPUT_FEATURE_STYLE[feature_names[i]] for i in feature_order],fontsize=30)
    ax.set_xlabel("mean (Shapley value) - (average impact on model output magnitude)",fontsize=30)
    plt.tight_layout()

def plot_shaply(model, X_test, class_labels, input_vars, plot_dir):
    """
    plots shap values for all classes and saves it in output directory 
    """
    labels = list(class_labels.keys())
    model2 = tf.keras.Model(model.input, model.output)
   
    for explainer, name  in [(shap.GradientExplainer(model2, X_test[:100]), "GradientExplainer"), ]:
        print("... {0}: explainer.shap_values(X)".format(name))
        #print(type(X_test[0,0,15]))
        shap_values = explainer.shap_values(X_test[:100])  # X_test[:1000]
        new = np.sum(shap_values, axis = 1)                # Sums all top particles of shap values
        # 15 is for dxy
        # print('shap values',shap_values.shape)
        # print('new',new.shape)
        # print('new',type(new))
        print("... shap summary_plot classification")
        plt.clf()
        new = np.transpose(new, (2,0,1))
        shapPlot(new, input_vars, labels)
        #plt.savefig(plot_dir+"/shap_summary_class.pdf",bbox_inches='tight')
        #plt.savefig(plot_dir+"/shap_summary_class.png",bbox_inches='tight')   pt=[0]
        plt.savefig(plot_dir+"/shap_test",bbox_inches='tight')

#creating a directory to save the plots
plot_dir = os.path.join("output", "plots/training")
os.makedirs(plot_dir, exist_ok=True)

#plot using the functions defined above
#Load the metadata for class_label
with open(f"training_data_CMS/class_label.json", 'r') as file: class_labels = json.load(file)
with open(f"training_data_CMS/input_vars.json", 'r') as file: input_vars = json.load(file)

ROC_dict = {class_label : 0 for class_label in class_labels}
#load test data
X_test = np.load(f"training_data_CMS/X_test.npy")
y_test = np.load(f"training_data_CMS/y_test.npy")
#print(type(X_test[1,1,0]))
#fig,ax=plt.subplots()

pt, counts = np.unique(X_test[:,1,0], return_counts=True)
print(np.max(X_test[:,1,0]))
plt.bar(pt,counts)
plt.xlim(0,2)
plt.savefig(plot_dir+"/pt_counts_test",bbox_inches='tight')
# #load saved model
# model = load_model(f"output/model/saved_model.h5")
# model_outputs = model.predict(X_test)
# y_pred = model_outputs
# print('model loaded')
# #ROC_dict = ROC(y_pred, y_test, class_labels, plot_dir,ROC_dict)
# #confusion(y_pred, y_test, class_labels, plot_dir)
# plot_shaply(model, X_test, class_labels, input_vars, plot_dir)
# print('done!')

# To run code on pixi
# /software/mm24721/jet_tagger_BSc/.pixi/envs/default/bin/python model_output.py