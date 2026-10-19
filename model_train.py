#Imports 

print('running model_train.py...')
import os, shutil, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from schema import Schema, And, Use, Optional
import shap
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import plot.style as style
import collections
import keras
from keras.layers import BatchNormalization,  Activation, AveragePooling1D, Dense, Conv1D, Flatten
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'figure.max_open_warning': 0})
setattr(collections, "MutableMapping", collections.abc.MutableMapping)
np.bool = np.bool_
num_threads = 24
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads) 
os.environ["TF_NUM_INTEROP_THREADS"] = str(num_threads)

tf.config.threading.set_inter_op_parallelism_threads(
    num_threads
)
tf.config.threading.set_intra_op_parallelism_threads(
    num_threads
)
tf.keras.utils.set_random_seed(420) # not a special number 

#functions for plotting
style.set_style()
def loss_history(plot_dir, history):
    """
    plots loss history of the model and saves in output directory 
    """
    for metric in ["loss",]:

        fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
        hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax, fontsize=style.CMSHEADER_SIZE)
        ax.plot(history.history[metric], label='Train Loss', linewidth=style.LINEWIDTH)
        ax.plot(history.history['val_'+metric], label='Validation Loss',linewidth=style.LINEWIDTH)
        ax.grid(True)
        # ax.set_ylabel('Loss')
        ax.set_ylabel('Loss '+metric)
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')

        save_path = os.path.join(plot_dir, "loss_"+metric+"_history")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')

        fig.clf()

# Load training data 
X_train = np.load('training_data_CMS/X_train.npy')
y_train = np.load('training_data_CMS/y_train.npy')

# define model
# DeepSet architecture 
def jet_tagger(input_shape,output_shape):
    L, C = input_shape
    inputs = keras.layers.Input(shape=input_shape, name='model_input')
    main = BatchNormalization(name='norm_input')(inputs)
    main = Conv1D(filters=10, kernel_size=1, name='Conv1D_1', activation='relu')(main)
    main = Conv1D(filters=10, kernel_size=1, name='Conv1D_2', activation='relu')(main)
    main = AveragePooling1D(L,name='avgpool')(main)
    main = Flatten()(main)
    jet_id = Dense(32, name='Dense_1' ,activation='relu')(main)
    jet_id = Dense(16, name='Dense_2' ,activation='relu')(jet_id)
    jet_id = Dense(8, name='Dense_3' )(jet_id)
    #jet_id = Activation('softmax', name='jet_id_output')(jet_id)
    jet_model = keras.Model(inputs = [inputs], outputs = [jet_id])
    return jet_model

input_shape = X_train.shape[1:] # First dimension is batch size
output_shape = y_train.shape[1:]
model = jet_tagger(input_shape, output_shape) 
callbacks = [
            EarlyStopping(monitor='val_loss', patience=10),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
            ),
        ]

model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics={'jet_id_output': 'categorical_accuracy'}
        )

# train
BATCH_SIZE = 512            #512
EPOCHS = 30                  #3
VALIDATION_SPLIT = 0.20     #0.2
history = model.fit( X_train,y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=2,
                            validation_split=VALIDATION_SPLIT,
                            callbacks = [callbacks],
                            shuffle=True)

#creating a directory to save the plots
plot_dir = os.path.join("output", "plots/training")
os.makedirs(plot_dir, exist_ok=True)
#loss function plot
loss_history(plot_dir,history)   

# save
export_path = os.path.join("output", "model/saved_model.h5")
model.save(export_path)
print('done!')
# To run code on pixi
# /software/mm24721/jet_tagger_BSc/.pixi/envs/default/bin/python model_train.py
