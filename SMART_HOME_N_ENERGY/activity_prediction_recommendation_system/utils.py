'''
input shape for build_model and build_model_usage is calculated as follows:
      build_model: len(aval_features)-1
      build_model_usage : len(usage_features)-1

--> needs to be set for every household individually
--> initialized for household 5
'''

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import keras_tuner as kt
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner import RandomSearch
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def prob(c):
    result = len([elem for elem in c if elem != 0])
    if result == 0:
        v = []
        for i in range(len(c)):
            v.append(0)
        return v
    else:
        """Compute probability values for each sets of scores in x."""
        return c / np.sum(c, axis=0)

def list_average(list):
    return sum(list)/len(list)


def build_model(hp):  
    model = keras.Sequential([
        keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=32, max_value=256, step=32),
            kernel_initializer=hp.Choice('kernel_initializer1',values =['normal','he_normal']), #'zero', 'glorot_normal',
            #kernel_initializer=keras.initializers.glorot_normal(seed=42), 
            bias_initializer='zeros',
            activation=hp.Choice('activation1',values=['softmax','sigmoid']),  #'relu',
            input_shape=(6-1,)
            ),
        #keras.layers.Dropout(hp.Float('dropout1',min_value = 0,max_value=0.6, step=0.2)),
        #keras.layers.Dense(
        #    units=hp.Int('dense_2_units', min_value=32, max_value=256, step=16),
        #    kernel_initializer=hp.Choice('kernel_initializer2',values =['normal','zero','glorot_normal','he_normal']),
        #    #kernel_initializer=keras.initializers.glorot_normal(seed=42), 
        #    bias_initializer='zeros',
        #    activation=hp.Choice('activation2',values=['softmax','relu','sigmoid']),
        #),
        #keras.layers.Dropout(hp.Float('dropout2',min_value = 0,max_value=0.6, step=0.2)),
            keras.layers.Dense(1, hp.Choice('activation4',values=['softmax', 'sigmoid'])) #'relu'
            ])
  
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-1,1e-2,1e-3,1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
  
    return model

def build_model_usage(hp):  
    model = keras.Sequential([
        keras.layers.Dense(
            units=hp.Int('dense_1_units', min_value=32, max_value=256, step=16),
            kernel_initializer=hp.Choice('kernel_initializer1',values =['normal','he_normal']), #'zero', 'glorot_normal'
            #kernel_initializer=keras.initializers.glorot_normal(seed=42), 
            bias_initializer='zeros',
            activation=hp.Choice('activation1',values=['softmax', 'sigmoid']), #'relu'
            input_shape=(30-1,)
        ),
        keras.layers.Dense(1, hp.Choice('activation4',values=['softmax', 'sigmoid'])) #'relu'
        ])
  
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-1,1e-2,1e-3,1e-4])),
              loss='binary_crossentropy',
              metrics=['accuracy']) 
  
    return model

def plot_dict(myDict,title,x_label,y_label,filename,line=0):
    myList = myDict.items()
    myList = sorted(myList) 
    x, y = zip(*myList) 
    
    if line is not 0:
        plt.axhline(line, color='tab:orange', linestyle='dashed')

    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.show()


def plot_dict_bar(myDict,title,x_label,y_label,filename,line=0):
    myList = myDict.items()
    myList = sorted(myList) 
    x, y = zip(*myList) 
    
    if line is not 0:
        plt.axhline(line, color='tab:orange', linestyle='dashed')
    
    plt.bar(x, y, width = 1, color = 'g')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.savefig(filename)
    plt.show()


def plot_AUC(y_test,y_pred):
    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label='(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.savefig('Log_ROC')
    plt.show()
