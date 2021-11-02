import pandas as pd
import numpy as np
import os
from scipy.io import arff
from scipy.spatial import distance
import math
from sklearn.base import BaseEstimator, TransformerMixin


class Activity_Agent(BaseEstimator,TransformerMixin):
    def __init__(self,num_devices,num_activities, activity_vector):
        #print('\n>>>>>init() activity called\n')
        self.num_devices = num_devices
        self.num_activities = num_activities
        self.activity_vector = activity_vector
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform activity called\n')
        data = X.copy()
        data = data.tail(24)
        
        # Get the activity-device mapping
        act = pd.read_csv(self.activity_vector, index_col=0, sep = ',')
        
        # Computing cosine similarity
        index = act.index.tolist()
        k=self.num_devices
        for n in index:
            data[n]=''
            for i in range(len(data)):
                # Vector indicating the device usage probabilities per hour
                vector1 = data.iloc[i:i+1,:self.num_devices].round(2)
                
                # Vector indicating the device usage per activity
                vector2 = act.loc[n,:].astype(float)
                
                # Compute cosine similarity between vector 1 and vector2
                cosine_similarity = 1 - distance.cosine(vector1, vector2)
                if math.isnan(cosine_similarity):
                    cosine_similarity = 0.0
                data.iloc[i:i+1,k] = cosine_similarity
            k+=1

        k=self.num_devices
        l = self.num_devices + self.num_activities
            
        # Compute the probabilities of an activity to be carried out per hour 
        for i in index:
            data[i+'_prob']=''
        for i in range(len(data)):
            # Vector of cosine similarity of all activities per hour
            vector = data.iloc[i:i+1,k:l]#.round(2) 
            vector = vector.values.tolist()[0]
            probs = prob(vector)
            for j in range(self.num_activities):
                data.iloc[i:i+1,l+j:l+j+1] = probs[j]
                
        return data
