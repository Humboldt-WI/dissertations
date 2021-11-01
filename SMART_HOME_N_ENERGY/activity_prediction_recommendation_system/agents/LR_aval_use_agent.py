import pandas as pd
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

class Availability_Agent_Logistic_Regression(BaseEstimator,TransformerMixin):
    def __init__(self,indices,split):
        #print('\n>>>>>init() Logistic_Regression_availability called\n')
        self.indices = indices
        self.split = split
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Logistic_Regression_availability called\n')
        data = X.copy()
        
        X = data.drop('User_aval',axis=1)
        y = data['User_aval'].values
        
        # Split data into training and test set
        train_indices, test_indices = self.indices[self.split:], self.indices[:self.split]
        
        X_train = X.loc[X.index[train_indices]]
        X_test = X.loc[X.index[test_indices]]
        X_test = X_test.append(X.iloc[-24:,:])

        y_train = y[train_indices]
        y_test = y[test_indices]
        y_test = np.append(y_test,y[-24:],axis=0)
        
        # Create and fit the logistic regression model
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        
        # Perform the prediction
        y_pred = logreg.predict_proba(X_test)[:, 1]
        
        # Evaluate the prediction
        auc_score = roc_auc_score(y_test, y_pred)
        #print('AUC of logistic regression classifier on test set: {:.2f}'.format(auc_score))

        return auc_score, y_pred, X_test


class Usage_Agent_Logistic_Regression(BaseEstimator,TransformerMixin):
    def __init__(self,num_devices, indices, split):
        #print('\n>>>>>init() Logistic_Regression_usage called\n')
        self.num_devices = num_devices
        self.indices = indices
        self.split = split
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Logistic_Regression_usage called\n')
        data = X.copy()
        
        # Get device usage columns
        columns = data.columns.tolist()[-self.num_devices:]  
        
        # Function to parallel the loop over the data columns indicating the usage of the devices 
        # to predict the device usage using a logistic regression model
        def LR_device(data,col):
            
            usage_pred = pd.DataFrame()
            auc_scores = pd.DataFrame()
        
            X = data.drop(col,axis=1)
            y = data[col].values
            
            # Split data into train and test sets
            train_indices, test_indices = self.indices[self.split:], self.indices[:self.split]

            X_train = X.loc[X.index[train_indices]]
            X_test = X.loc[X.index[test_indices]]
            X_test = X_test.append(X.iloc[-24:,:])
            
            y_train = y[train_indices]
            y_test = y[test_indices]
            y_test = np.append(y_test,y[-24:],axis=0)
            
            # Create and fit the logistic regression model
            logreg = LogisticRegression(solver = 'liblinear')
            logreg.fit(X_train, y_train)
    
            # Perform the prediction
            y_pred = logreg.predict_proba(X_test)[:,1]
            
            # Evaluate the prediction
            auc = roc_auc_score(y_test, y_pred)

            usage_pred[col] = y_pred
            auc_scores = auc_scores.append({col:auc},ignore_index = True)
            
            return usage_pred, auc_scores
        
        # Call the LR_device function to get the usage prediction and its evaluation for each device 
        results = Parallel(n_jobs=29)(delayed(LR_device)(data, col) for col in columns)
        
        # Convert the format of the results
        flat = []
        for l in results:
            flat.extend(l)
        
        result1 = []
        result2 = []
        for i in range(len(flat)):
            if i==0 or i%2==0:
                result1.append(flat[i])
            else:
                result2.append(flat[i])
    
        usage_pred = pd.concat(result1,axis=1)
        auc_scores = pd.concat(result2,axis=1)
        
        return auc_scores, usage_pred


