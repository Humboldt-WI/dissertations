import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from sklearn.base import BaseEstimator, TransformerMixin

class hourly_grouping(BaseEstimator, TransformerMixin):
    def __init__(self):
        #print('\n>>>>>init() grouping called\n')
        return 
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None): 
        #print('\n>>>>>transform grouping called\n')
        data = X.copy()
        
        # Drop Unix and Aggregate consumption columns
        data = data.iloc[:,2:11]
        
        data['Date'] = data.index.to_frame()
        data = data.assign(Date=data.Date.dt.round('H')- timedelta(hours=1))
        data['Hour'] = data['Date'].dt.hour
        data = data.groupby(['Date','Hour']).mean()
        data = data.reset_index(level=['Hour'])
        data = data.astype(int)
        
        return data

class time_features(BaseEstimator, TransformerMixin):
    def __init__(self):
        #print('\n>>>>>init() time_features called\n')
        return 
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform time_features called\n')
        data = X.copy()
        data['Date'] = data.index.to_frame()
        data['Month'] = pd.DatetimeIndex(data['Date']).month
        data['Day_of_week'] = data['Date'].dt.day_name()
        
        data = data.drop(['Date'],axis=1)
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Use ordered cat.codes 
        data['Day_of_week'] = data['Day_of_week'].astype('category')
        data['Day_of_week'] = data['Day_of_week'].cat.reorder_categories(day_order, ordered=True)
        data['Day_of_week'] = data['Day_of_week'].cat.codes.astype(int)
        
        # Change order of columns
        cols = data.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        data = data[cols]
        
        return data

class appliance_label_house5(BaseEstimator,TransformerMixin):
    def __init__(self):
        #print('\n>>>>>init() appliance_label_house5 called\n')
        return
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform app_labels called\n')
        data = X.copy()
        
        # Rename columns 
        data = data.rename(columns={   
            "Appliance1":"Fridge-Freezer",
            "Appliance2":"Tumble_Dryer",
            "Appliance3":"Washing_machine",
            "Appliance4":"Dishwasher",
            "Appliance5":"Desktop_Computer",
            "Appliance6":"Television_Site",
            "Appliance7":"Microwave",
            "Appliance8":"Kettle",
            "Appliance9":"Toaster"
            })
        
        return data

class appliance_label_house1(BaseEstimator,TransformerMixin):
    def __init__(self):
        #print('\n>>>>>init() appliance_label_house1 called\n')
        return
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform app_labels called\n')
        data = X.copy()
        
        # Rename columns 
        data = data.rename(columns={   
            "Appliance1":"Fridge",
            "Appliance2":"Freezer_1",
            "Appliance3":"Freezer_2",
            "Appliance4":"Washer_Dryer",
            "Appliance5":"Washing_Machine",
            "Appliance6":"Dishwasher",
            "Appliance7":"Computer",
            "Appliance8":"Television",
            "Appliance9":"Electric_Heater"
            })
        
        return data

class appliance_label_house2(BaseEstimator,TransformerMixin):
    def __init__(self):
        #print('\n>>>>>init() appliance_label_house2 called\n')
        return
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform app_labels called\n')
        data = X.copy()
        
        # Rename columns 
        data = data.rename(columns={   
            "Appliance1":"Fridge-Freezer",
            "Appliance2":"Washing_Machine",
            "Appliance3":"Dishwasher",
            "Appliance4":"Television",
            "Appliance5":"Microwave",
            "Appliance6":"Toaster",
            "Appliance7":"Hi-Fi",
            "Appliance8":"Kettle",
            "Appliance9":"Overhead_Fan"
            })
        
        return data

class appliance_label_house3(BaseEstimator,TransformerMixin):
    def __init__(self):
        #print('\n>>>>>init() appliance_label_house3 called\n')
        return
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform app_labels called\n')
        data = X.copy()
        
        # Rename columns 
        data = data.rename(columns={   
            "Appliance1":"Toaster",
            "Appliance2":"Fridge_Freezer",
            "Appliance3":"Freezer",
            "Appliance4":"Tumble_Dryer",
            "Appliance5":"Dishwasher",
            "Appliance6":"Washing_Machine",
            "Appliance7":"Television_Site",
            "Appliance8":"Microwave",
            "Appliance9":"Kettle"
            })
        
        return data

class appliance_label_house4(BaseEstimator,TransformerMixin):
    def __init__(self):
        #print('\n>>>>>init() appliance_label_house4 called\n')
        return
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform app_labels called\n')
        data = X.copy()
        
        # Rename columns 
        data = data.rename(columns={   
            "Appliance1":"Fridge",
            "Appliance2":"Freezer",
            "Appliance3":"Fridge-Freezer",
            "Appliance4":"Washing_Machine(1)",
            "Appliance5":"Washing_Machine(2)",
            "Appliance6":"Computer",
            "Appliance7":"Television_Site",
            "Appliance8":"Microwave",
            "Appliance9":"Kettle"
            })
        
        return data

class availability(BaseEstimator,TransformerMixin):
    def __init__(self,on_threshold):
        #print('\n>>>>>init() availability called\n')
        self.on_threshold = on_threshold
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform availability called\n')
        data = X.copy()
        
        # Hourly user availability due to usage of availability devices - 0: no availability, 1: availability
        data.insert(3,'User_aval','')
        
        columns = data.columns.tolist()
        
        # Drop devices that are not shiftable
        for col in columns:
            if col in (['Fridge_Freezer','Freezer','Freezer_1','Freezer_2','Fridge','Router','Fridge-Freezer','Overhead_Fan','Electric_Heater']):
                data = data.drop([col],axis=1)

        for col in columns:
            if col in (['Microwave','Kettle','Toaster','Hi-Fi','Television']):  # Appliances indicating availability
                for i in range(len(data)):
                    if data.loc[data.index[i],'User_aval']!=1:
                        if data.loc[data.index[i],col]>self.on_threshold:
                            data.loc[data.index[i],'User_aval']=1
                        else: 
                            data.loc[data.index[i],'User_aval']=0       
        
        # Availability lag of 1 hour 
        data.insert(3,'User_aval_t-1','')
        data['User_aval_t-1'] = data['User_aval'].shift(1)
        
        # Availability lag of 1 week (7*24 hours) 
        data.insert(3,'User_aval_t-1week','')
        data['User_aval_t-1week'] = data['User_aval'].shift(168)
        
        return data

class device_usage(BaseEstimator,TransformerMixin):
    def __init__(self,num_devices,on_threshold):
        #print('\n>>>>>init() device_usage called\n')
        self.num_devices = num_devices
        self.on_threshold = on_threshold
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform device_usage called\n')
        data = X.copy()
        columns = data.columns.tolist()[-self.num_devices:]
        
        # Check if device is in use (if greater than on_threshold)
        for col in columns:
            data[col+'_usage']=''
            for i in range(len(data)):
                if data.loc[data.index[i],col]>self.on_threshold:  
                    data.loc[data.index[i],col+'_usage']=1
                else:
                    data.loc[data.index[i],col+'_usage']=0
                    
        columns = data.columns.tolist()[-self.num_devices:]
            
        # Device usage lag 1 hour
        for col in columns:
            data[col+'_t-1'] = data[col].shift(1)
        
        # Device usage lag 1 week
        for col in columns:
            data[col+'_t-1week'] = data[col].shift(168)
        
        # Delete first week due to incomplete data
        data = data.iloc[168:,:]
        
        data = data.round()
        data = data.astype(int)
            
        return data

class FeatureSelector( BaseEstimator, TransformerMixin ):
    def __init__(self, feature_names):
        #print('\n>>>>>init() FeatureSelector called\n')
        self.feature_names = feature_names 
      
    def fit( self, X, y = None ):
        return self 
    
    def transform( self, X, y = None ):
        #print('\n>>>>>transform() FeatureSelector called\n')
        return X[self.feature_names] 
