import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Activity_Agent_Evaluation(BaseEstimator,TransformerMixin):
    def __init__(self,usage_threshold,activity_threshold,num_devices,num_activities, activity_vector):
        self.usage_threshold = usage_threshold
        self.activity_threshold = activity_threshold
        self.num_devices = num_devices
        self.num_activities = num_activities
        self.activity_vector = activity_vector
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        data = X.copy()
        data = data.tail(24)
        data = data.round(3)
        
        l = self.num_devices+self.num_activities
        
        # Get the activity-device mapping
        act_vec = pd.read_csv(self.activity_vector, index_col=0, sep = ',')

        # Creation of a dictionary with activities as keys and their identifying devices as corresponding values
        ident_dev_act={}
        for i in range(len(act_vec)):
            act = act_vec.index[i]
            prob = act_vec.columns[act_vec.loc[act].eq(1)].tolist()
            if len(prob)==1:
                prob = prob[0]
            ident_dev_act.update({act:prob})       
        ident_dev_act

        # Count of total comparisons
        total = 0
        # Count of right prediction sets
        count = 0
        for i in range(24):
            activity_list = list(data.iloc[i:i+1,self.num_devices:l].columns)
            list_dev_keys = []
            list_act_prob = []
            no_match = False
    
            # Get the activities that would result from the devices with a higher usage probability
            # depending on the usage_threshold
            for j in range(self.num_devices):
                use_prob = data.iloc[i:i+1,:self.num_devices].sort_values(by = data.index[i],axis=1,ascending = False).transpose().iloc[j:j+1,:].values.item()
                if use_prob>self.usage_threshold:
                    dev = data.iloc[i:i+1,:self.num_devices].sort_values(by = data.index[i],axis=1,ascending = False).transpose().iloc[j:j+1,:].index.values
                    dev = ''.join(map(str,dev)).replace('_usage','')
                    
                    # activities that are identified by more than one device
                    # House 5
                    if dev in (['Tumble_Dryer','Washing_machine']):
                        dev = ['Tumble_Dryer','Washing_machine']
                    if dev in (['Microwave','Kettle','Toaster']):
                        dev = ['Microwave','Kettle','Toaster'] 
                    
                    # House 1
                    #if dev in ['Washer_Dryer', 'Washing_Machine']:
                    #    dev = ['Washer_Dryer', 'Washing_Machine']
                    
                    # House 2
                    #if dev in ['Microwave', 'Toaster', 'Kettle']:
                    #    dev = ['Microwave', 'Toaster', 'Kettle']
                    #if dev in ['Television', 'Hi-Fi']:
                    #    dev = ['Television', 'Hi-Fi'] 
                        
                    # House 3
                    #if dev in ['Tumble_Dryer', 'Washing_Machine']:
                    #    dev = ['Tumble_Dryer', 'Washing_Machine']
                    #if dev in ['Toaster', 'Microwave', 'Kettle']:
                    #    dev = ['Toaster', 'Microwave', 'Kettle']
                        
                    # House 4
                    #if dev in ['Washing_Machine(1)', 'Washing_Machine(2)']:
                    #    dev = ['Washing_Machine(1)', 'Washing_Machine(2)']
                    #if dev in ['Microwave', 'Kettle']:
                    #    dev = ['Microwave', 'Kettle']

                    # Get the activity that uses the devices as identifying one(s)
                    act = [k for k, v in ident_dev_act.items() if v == dev][0]
            
                    if act in activity_list:
                        list_dev_keys.append(act)
                        activity_list.remove(act)
    
            # Get the activities the Activity Agent predicted
            # depending on the activity_threshold
            for j in range(self.num_activities):
                act_prob = data.iloc[i:i+1,l:].sort_values(by = data.index[i],axis=1,ascending = False).transpose().iloc[j:j+1,:].values.item()
                if act_prob>self.activity_threshold:
                    act_prob = data.iloc[i:i+1,l:].sort_values(by = data.index[i],axis=1,ascending = False).transpose().iloc[j:j+1,:].index.values
                    act_prob = ''.join(map(str,act_prob)).replace('_prob','')
        
                    list_act_prob.append(act_prob)
    
            # Check if both lists contain equal elements
            check =  any(item in list_dev_keys for item in list_act_prob) #all
 
            if check is True and len(list_dev_keys) and len(list_act_prob):
                count+=1
                total+=1
            else:
                total+=1

        # Compute EQUAL score (ratio of equal sets to total number of sets)
        EQUAL = count/total
        
        return EQUAL
