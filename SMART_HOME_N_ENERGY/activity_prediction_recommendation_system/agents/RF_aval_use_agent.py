'''
Disable calculation of AUC scores for running the overall recommendation system
to prevent ValueError: Only one class present in y_true. ROC AUC score is not defined in that case. 
due to the adding of missing hourly data for a complete dataset which is necessary
for the Recommendation Agent
'''

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

class Availability_Agent_First_Random_Forest(BaseEstimator,TransformerMixin):
    def __init__(self,indices,split):
        #print('\n>>>>>init() First_Random_Forest_availability called\n')
        self.indices = indices
        self.split = split
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Random_Forest_availability called\n')
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
        
        # Create and fit the random forest model
        rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
        rf.fit(X_train, y_train)
        
        # Perform the prediction
        y_pred = rf.predict_proba(X_test)[:, 1]
        
        # Evaluate the prediction
        auc_score = roc_auc_score(y_test, y_pred)
        
        # Get the current hyperparameters
        params_used = rf.get_params()
        
        return auc_score, params_used


class Availability_Agent_Random_Forest(BaseEstimator,TransformerMixin):
    def __init__(self,indices,split,params,boolean,random_grid):
        #print('\n>>>>>init() Random_Forest_availability called\n')
        self.indices = indices
        self.split = split
        self.params = params
        self.boolean = boolean
        self.random_grid = random_grid
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Random_Forest_availability called\n')
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
        
        # if boolean = True tune the model's hyperparameters
        # if boolean = False train the model with the current hyperparameters
        if self.boolean:
            # Create and fit the model
            rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(),
                                           param_distributions = self.random_grid, 
                                           n_iter = 50, 
                                           cv = 3, 
                                           verbose=2, 
                                           random_state=42, 
                                           # Number of jobs to run in parallel: -1 --> using all processors
                                           n_jobs = -1)
            rf_random.fit(X_train, y_train)
            
            # Perform the prediction
            y_pred = rf_random.best_estimator_.predict_proba(X_test)[:,1]
            
            # Evaluate the prediction
            auc_score = roc_auc_score(y_test, y_pred)
            #print('AUC of tuned random forest classifier on test set: {:.2f}'.format(roc_auc_score(y_test, y_pred)))
            
            # Get the best hyperparameters
            params_used = rf_random.best_params_
            print('Best parameters:\n')
            pprint(params_used)
            
        else:
            # Create and fit the random forest model
            rf = RandomForestClassifier(n_estimators=self.params['n_estimators'],
                                        max_features=self.params['max_features'],
                                        max_depth=self.params['max_depth'],
                                        min_samples_split=self.params['min_samples_split'],
                                        min_samples_leaf=self.params['min_samples_leaf'],
                                        bootstrap=self.params['bootstrap'],
                                        random_state = 42)
        
            rf.fit(X_train, y_train)
            
            # Perform the prediction
            y_pred = rf.predict_proba(X_test)[:, 1]
            
            # Evaluate the prediction
            auc_score = roc_auc_score(y_test, y_pred)
            #print('AUC of random forest classifier on test set: {:.2f}'.format(auc_score))
            
            # Get current hyperparameters
            params_used = rf.get_params()
            #print('Parameters currently in use:\n')
            #pprint(params_used
        
        return auc_score, params_used, y_pred, X_test


class Usage_Agent_Random_Forest_tuning(BaseEstimator,TransformerMixin):
    def __init__(self,num_devices,indices,split,random_grid):
        self.num_devices = num_devices
        self.indices = indices
        self.split = split
        self.random_grid = random_grid
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        data = X.copy()
        
        # Get device usage columns
        columns = data.columns.tolist()[-self.num_devices:] 
        
        params_dict = {}
        auc_scores = pd.DataFrame(columns = columns)
        usage_pred = pd.DataFrame(columns = columns)
        
        for col in columns:
            
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
            
            # Create and fit the random forest model
            rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(),
                                          param_distributions = self.random_grid, 
                                           n_iter = 50, 
                                           cv = 3, 
                                           verbose=2, 
                                           random_state=42, 
                                           # Number of jobs to run in parallel: -1 --> using all processors
                                           n_jobs = 4) # number of concurrent workers
            
            rf_random.fit(X_train, y_train)
        
            # Perform the prediction    
            y_pred = rf_random.best_estimator_.predict_proba(X_test)[:,1]
            
            # Evaluate the prediction
            auc_score = roc_auc_score(y_test, y_pred)
            
            # Get the best hyperparameters
            params_used = rf_random.best_params_
            #print('Best parameters:\n')
            #pprint(params_used)
            
            params_dict.update({col:params_used})
            auc_scores.loc[0,col] = auc_score
            usage_pred[col] = y_pred
            
        return auc_scores, usage_pred, params_dict


class Usage_Agent_Random_Forest(BaseEstimator,TransformerMixin):
    def __init__(self,num_devices,indices, split, random_grid,params_dict):
        #print('\n>>>>>init() Random_Forest_usage called\n')
        self.num_devices = num_devices
        self.indices = indices
        self.split = split
        self.random_grid = random_grid
        self.params_dict = params_dict
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Random_Forest_usage called\n')
        data = X.copy()
        
        # Get device usage columns
        columns = data.columns.tolist()[-self.num_devices:]  
            
        # Function to parallel the loop over the data columns indicating the usage of the devices 
        # to predict the device usage using a random forest model   
        def RF_device(data,col):
            
            import pandas as pd
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score
            
            auc_scores = pd.DataFrame()
            usage_pred = pd.DataFrame()
            
            # Split data into train and test sets
            X = data.drop(col,axis=1)
            y = data[col].values
    
            train_indices, test_indices = self.indices[self.split:], self.indices[:self.split]
        
            X_train = X.loc[X.index[train_indices]]
            X_train = X_train.append(X.iloc[:24*21,:])
            X_test = X.loc[X.index[test_indices]]
            X_test = X_test.append(X.iloc[-24:,:])
        
            y_train = y[train_indices]
            y_train = np.append(y_train,y[:24*21],axis=0)
            y_test = y[test_indices]
            y_test = np.append(y_test,y[-24:],axis=0)
            params = self.params_dict.get(col)
            
            # Create and fit the random forest model    
            rf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                        max_features=params['max_features'],
                                        max_depth=params['max_depth'],
                                        min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        bootstrap=params['bootstrap'],
                                        random_state = 42)
        
            rf.fit(X_train, y_train)
            
            # Perform the prediction
            y_pred = rf.predict_proba(X_test)[:, 1]
            
            # Evaluate the prediction
            auc = roc_auc_score(y_test, y_pred)
                
            usage_pred[col] = y_pred#[-24:]
            auc_scores = auc_scores.append({col:auc},ignore_index = True)
                
            return usage_pred, auc_scores
        
        # Call the RF_device function to get the usage prediction and its evaluation for each device 
        results = Parallel(n_jobs=29,backend='threading',verbose=2)(delayed(RF_device)(data, col) for col in columns)
        
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
