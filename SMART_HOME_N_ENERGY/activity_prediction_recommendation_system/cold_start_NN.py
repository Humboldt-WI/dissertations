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

class Availability_Agent_Neural_Network_cold_start(BaseEstimator,TransformerMixin):
    def __init__(self, day,start,stop, model,cut):
        #print('\n>>>>>init() Neural_Network_availability called\n')
        self.day = day
        self.start = start
        self.stop = stop
        self.model = model
        self.cut = cut
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Neural_Network_availability called\n')
        data = X.copy()
        
        # Split data into training and test set
        train = data.iloc[self.cut:24*self.day,:]
        test = data.iloc[self.start*24:self.stop*24,:]

        train = train.sample(frac=1).reset_index(drop=True)
        test = test.sample(frac=1).reset_index(drop=True)

        X_train = train.drop('User_aval',axis=1)
        X_test = test.drop('User_aval',axis=1)
        
        y_train = train['User_aval'].values
        y_test = test['User_aval'].values
       
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
        
        # Evaluate the prediction
        auc_score = roc_auc_score(y_test, y_pred)
        
        return auc_score

class Availability_Agent_Neural_Network_tuning_cold_start(BaseEstimator,TransformerMixin):
    def __init__(self,day,cut):
        #print('\n>>>>>init() Neural_Network_availability called\n')
        self.day = day
        self.cut = cut
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Neural_Network_availability called\n')
        data = X.copy()
        seed = 42
        
        # Split data into training and test set
        train = data.iloc[self.cut:24*self.day,:]
        train = train.sample(frac=1).reset_index(drop=True)

        X_train = train.drop('User_aval',axis=1)
        y_train = train['User_aval'].values
        
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
                          directory='output',project_name="Aval",)

      
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

class Usage_Agent_Neural_Network_tuning_cold_start(BaseEstimator,TransformerMixin):
    def __init__(self,num_devices,day,cut):
        #print('\n>>>>>init() Neural_Network_usage_tuning called\n')
        self.num_devices = num_devices
        self.day = day
        self.cut = cut
        
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
            
            train = data.iloc[self.cut:24*self.day,:]
            
            X_train = train.drop(col,axis=1)
            y_train = train[col].values

            # Normalize training data
            scaler = MinMaxScaler(feature_range = (0,1))
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)

            # Initialize the Hyperband Tuner 
            tuner_search=kt.Hyperband(utils.build_model_usage,
                          objective='val_accuracy',
                          max_epochs=20,
                          factor=3,
                          overwrite = True,
                          #directory='output_'+str(col),
                          directory=os.path.normpath('C:/Users/loeschml/Documents'),
                          project_name=str(col))

            # Stop training if the "val_loss" has not improved in 5 epochs
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            tuner_search.search(X_train_scaled,y_train,epochs=50,validation_split=0.2,callbacks=[stop_early])
        
            # Save the best hyperparameters
            best_hps=tuner_search.get_best_hyperparameters(num_trials=1)[0]
        
            # Get the best model
            model=tuner_search.get_best_models(num_models=1)[0]

            # Get the summary of the model
            model.summary()
        
            best_hps_dict.update({col:best_hps})
            model_dict.update({col:model})
            
        
        return model_dict

class Usage_Agent_Neural_Network_cold_start(BaseEstimator,TransformerMixin):
    def __init__(self,num_devices,day,start,stop,model_dict,cut):
        #print('\n>>>>>init() Neural_Network_usage called\n')
        self.num_devices = num_devices
        self.day = day
        self.model_dict = model_dict
        self.start = start
        self.stop = stop
        self.cut = cut
        
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
            
            usage_pred = pd.DataFrame()
            auc_scores = pd.DataFrame()
            
            # Split data into training and test set
            train = data.iloc[self.cut:24*self.day,:]
            test = data.iloc[24*self.start:24*self.stop,:]

            X_train = train.drop(col,axis=1)
            X_test = test.drop(col,axis=1)
            
            y_train = train[col].values
            y_test = test[col].values
            
            # Normalize the training set
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
            
            usage_pred[col] = y_pred
            auc_scores = auc_scores.append({col:auc},ignore_index = True)
                
            return usage_pred, auc_scores
        
       
        # Call the NN_device function to get the usage prediction and its evaluation for each device 
        results = Parallel(n_jobs=29,backend='threading',verbose=2)(delayed(NN_device)(data,col) for col in columns)
        
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
        
        return auc_scores, usage_pred
