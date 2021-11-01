'''
Disable calculation of AUC scores for running the overall recommendation system
to prevent ValueError: Only one class present in y_true. ROC AUC score is not defined in that case. 
due to the adding of missing hourly data for a complete dataset
'''

import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import keras_tuner as kt
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner import RandomSearch
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import utils

class tuner_search(kt.Hyperband):
      def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        # via overriding `run_trial`
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=16)
        #Hyperband sets the epochs to train for via its own logic, so if you're using Hyperband you shouldn't tune the epochs
        #kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
        super(tuner_search, self).run_trial(trial, *args, **kwargs)


class Availability_Agent_Neural_Network(BaseEstimator,TransformerMixin):
    def __init__(self, indices, split, model):
        #print('\n>>>>>init() Neural_Network_availability called\n')
        self.indices = indices
        self.split = split
        self.model = model
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Neural_Network_availability called\n')
        data = X.copy()
        
        X = data.drop('User_aval',axis=1)
        y = data['User_aval'].values
        
        # Split data into training and test set
        train_indices, test_indices = self.indices[self.split:], self.indices[:self.split]

        X_train = X.loc[X.index[train_indices]]
        X_train = X_train.append(X.iloc[:24*21,:])
        X_test = X.loc[X.index[test_indices]]
        X_test = X_test.append(X.iloc[-24:,:])
        
        y_train = y[train_indices]
        y_train = np.append(y_train,y[:24*21],axis=0)
        y_test = y[test_indices]
        y_test = np.append(y_test,y[-24:],axis=0)
        
        # Normalize the data
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and fit the neural network model 
        model = self.model
        model.fit(X_train_scaled, y_train,
                    batch_size=20,
                    epochs=50,
                    verbose=0,
                    validation_data=(X_test_scaled, y_test))
    
        # Perform the prediction
        y_pred = model.predict(X_test_scaled)
        y_pred = np.squeeze(y_pred)
        
        auc_score = roc_auc_score(y_test, y_pred)
        
        return   auc_score, y_pred, X_test 

class Availability_Agent_Neural_Network_tuning(BaseEstimator,TransformerMixin):
    def __init__(self, indices, split):
        #print('\n>>>>>init() Neural_Network_availability called\n')
        self.indices = indices
        self.split = split
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Neural_Network_availability called\n')
        data = X.copy()
        seed = 42
        
        X = data.drop('User_aval',axis=1)
        y = data['User_aval'].values
    
        # Split data into training and test set
        train_indices, test_indices = self.indices[self.split:], self.indices[:self.split]

        X_train = X.loc[X.index[train_indices]]
        X_train = X_train.append(X.iloc[:24*21,:])
        y_train = y[train_indices]
        y_train = np.append(y_train,y[:24*21],axis=0)
        
        # Normalize the training set
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        
        # Initialize the Hyperband Tuner 
        tuner_search=kt.Hyperband(utils.build_model,
                          objective='val_accuracy',
                          max_epochs=10,
                          factor=3,
                          overwrite = True,
                          directory='output',project_name="Aval")

        # Stop training if the "val_loss" has not improved in 5 epochs
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner_search.search(X_train_scaled,y_train,epochs=50,validation_split=0.2,callbacks=[stop_early])
        
        # Save best hyperparameters
        best_hps=tuner_search.get_best_hyperparameters(num_trials=1)[0]
        
        # Get the best model
        model=tuner_search.get_best_models(num_models=1)[0]

        # Get the summary of the model
        model.summary()
        
        return model

class Usage_Agent_Neural_Network_tuning(BaseEstimator,TransformerMixin):
    def __init__(self,num_devices,indices, split):
        #print('\n>>>>>init() Neural_Network_usage_tuning called\n')
        self.num_devices = num_devices
        self.indices = indices
        self.split = split
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Neural_Network_usage_tuning called\n')
        data = X.copy()
        
        # Get device usage columns
        columns = data.columns.tolist()[-self.num_devices:] 
        
        best_hps_dict = {}
        model_dict = {}
        
        for col in columns:

            X = data.drop(col,axis=1)
            y = data[col].values
            
            # Split data into training and test set
            train_indices, test_indices = self.indices[self.split:], self.indices[:self.split]

            X_train = X.loc[X.index[train_indices]]
            X_train = X_train.append(X.iloc[:24*21,:])
            
            y_train = y[train_indices]
            y_train = np.append(y_train,y[:24*21],axis=0)
            
            # Normalize the training set
            scaler = MinMaxScaler(feature_range = (0,1))
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
        
            # Initialize the Hyperband Tuner 
            tuner_search=kt.Hyperband(utils.build_model_usage,
                          objective='val_accuracy',
                          max_epochs=10,
                          factor=3,
                          overwrite = True,
                          #directory='output_'+str(col),
                          directory=os.path.normpath('C:/Users/loeschml/Documents'),
                          project_name=str(col))

            # Stop training if the "val_loss" has not improved in 5 epochs
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            tuner_search.search(X_train_scaled,y_train,epochs=50,validation_split=0.2,callbacks=[stop_early])
        
            # Save best hyperparameters
            best_hps=tuner_search.get_best_hyperparameters(num_trials=1)[0]
        
            # Get the best model
            model=tuner_search.get_best_models(num_models=1)[0]

            # Get the summary of the model
            model.summary()
        
            best_hps_dict.update({col:best_hps})
            model_dict.update({col:model})
        
        return model_dict

class Usage_Agent_Neural_Network(BaseEstimator,TransformerMixin):
    def __init__(self,num_devices,indices,split, model_dict):
        #print('\n>>>>>init() Neural_Network_usage called\n')
        self.num_devices = num_devices
        self.indices = indices
        self.split = split
        self.model_dict = model_dict
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Neural_Network_usage called\n')
        data = X.copy()
        
        # Get device usage columns
        columns = data.columns.tolist()[-self.num_devices:]  
        
        # Function to parallel the loop over the data columns indicating the usage of the devices 
        # to predict the device usage using a neural network model
        def NN_device(data,col):
            
            import pandas as pd
            import numpy as np
            from sklearn.metrics import roc_auc_score
            from joblib import parallel_backend
            
            usage_pred = pd.DataFrame()
            auc_scores = pd.DataFrame()
            
            X = data.drop(col,axis=1)
            y = data[col].values
            
            # Split data into training and test set
            train_indices, test_indices = self.indices[self.split:], self.indices[:self.split]

            X_train = X.loc[X.index[train_indices]]
            X_train = X_train.append(X.iloc[:24*21,:])
            X_test = X.loc[X.index[test_indices]]
            X_test = X_test.append(X.iloc[-24:,:])
            
            y_train = y[train_indices]
            y_train = np.append(y_train,y[:24*21],axis=0)
            y_test = y[test_indices]
            y_test = np.append(y_test,y[-24:],axis=0)
            
            # Normalize the data
            scaler = MinMaxScaler(feature_range = (0,1))
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create and fit the model
            batch_size = 64 
            epochs = 80
            model = self.model_dict.get(col)
            model.fit(X_train_scaled, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=0,
                              validation_split=0.2)
            
            # Perform the prediction
            y_pred = model.predict(X_test_scaled)
            y_pred = np.squeeze(y_pred)
            
            # Evaluate the model
            auc = roc_auc_score(y_test, y_pred)
            #print('AUC of neural network on test set: {:.2f}'.format(roc_auc_score(y_test, y_pred)))
            
            usage_pred[col] = y_pred
            auc_scores = auc_scores.append({col:auc},ignore_index = True)
            
            return usage_pred, auc_scores  # Disable the AUC calculation by using data as a placeholder for auc_scores
                                           # --> 'return usage_pred, data' instead of 'return usage_pred, auc_scores'
        
       
        # Call the NN_device function to get the usage prediction and its evaluation for each device 
        results = Parallel(n_jobs=4,backend='threading',verbose=2)(delayed(NN_device)(data,col) for col in columns)
        
        # Convert the format of the results
        flat = []
        for l in results:
            flat.extend(l)
        
        result1 = []
        result2 = []
        for i in range(len(flat)):
            if i%2==0:
                result1.append(flat[i])
            else:
                result2.append(flat[i])
    
    
        usage_pred = pd.concat(result1,axis=1)
        auc_scores = pd.concat(result2,axis=1)
        
        return  auc_scores, usage_pred
