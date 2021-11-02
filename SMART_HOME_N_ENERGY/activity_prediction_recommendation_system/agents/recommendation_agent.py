import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import collections
from collections import ChainMap
from sklearn.preprocessing import MinMaxScaler

class Recommendation_Agent(BaseEstimator, TransformerMixin):
    def __init__(self,num_activities,num_devices, aval_off,emissions_ratio,activity_vector,dev_avg_con,aval_threshold, activity_threshold):
        #print('\n>>>>>init() Recommendation called\n')
        self.aval_threshold = aval_threshold
        self.activity_threshold = activity_threshold
        self.num_activities = num_activities
        self.num_devices = num_devices
        self.aval_off = aval_off
        self.emissions_ratio = emissions_ratio
        self.activity_vector = activity_vector
        self.dev_avg_con = dev_avg_con
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        #print('\n>>>>>transform Recommendation called\n')
        data = X.copy()

        # Get activity-device mapping
        act_vec = pd.read_csv(self.activity_vector,index_col=0, sep = ',')
        
        # Creation of a dictionary with activities as keys and their identifying devices as corresponding values
        ident_dev_act={}
        for i in range(len(act_vec)):
            act = act_vec.index[i]
            prob = act_vec.columns[act_vec.loc[act].eq(1)].tolist()
            if len(prob)==1:
                prob = prob[0]
            ident_dev_act.update({act:prob}) 
        
        # Recommendation horizon
        print('\n\033[1mRecommendation for {0} to {1}:\033[0m'.format(data.first_valid_index(),data.last_valid_index()))
        
        # Greenest hour of the day
        greenest_hour_ci = data['avg_intensity_forecast'].idxmin().hour
        print('\033[1m\nGreenest hour of the day: {0}\033[0m'.format(greenest_hour_ci))
        intensity_ci = data['avg_intensity_forecast'].min()#.item()
        print('Lowest hourly carbon intensity today: {0} gCO2/KWh\n'.format(round(intensity_ci,0)))
        
        # Cheapest hour of the day
        cheapest_hour_price = data['price'].idxmin().hour
        print('\033[1m\nCheapest hour of the day: {0}\033[0m'.format(cheapest_hour_price))
        intensity_price = data['price'].min()#.item()
        print('Lowest hourly price today: {0} €/KWh\n'.format(round(intensity_price,2)))
        
        # Get hours of predicted availability depending on the availability threshold
        aval_forecast = data['aval']
        aval = aval_forecast[aval_forecast>self.aval_threshold]
        aval_hours = aval.index.hour.values.tolist()
        
        #if len(aval_hours)!=0:
        #    print('Predicted hours of availability:')
        #    print(aval_hours)
        #    print('\n')
        
        data['Date'] = data.index.to_frame()
        data['Hour'] = data['Date'].dt.hour
        data = data.drop(columns=['Date'],axis=1)
        
        # Get predicted device usage per hour for the calculation of the energy consumption per activity
        dev_use = data.iloc[:,-(self.num_devices+1):-1]
        for i in range(len(dev_use)):
            for j in range(len(dev_use.columns)):
                if dev_use.iloc[i,j]> 0.2:
                    dev_use.iloc[i,j]=1
                else:
                    dev_use.iloc[i,j]=0
        dev_use = dev_use.astype(int)
        dev_use['Hour']=data['Hour']
                   
        # Get predicted activities and their predicted hours depending on the activity threshold
        act24 = data.iloc[:,-(self.num_activities+self.num_devices+1):-(self.num_devices+1)]
        for i in range(len(act24)):
            for j in range(len(act24.columns)):
                if act24.iloc[i,j]>self.activity_threshold:
                    act24.iloc[i,j]=1
                else:
                    act24.iloc[i,j]=0
        act24 = act24.astype(int)
        act24['Hour']=data['Hour']

        
        # Get the duration of predicted activities
        columns = act24.iloc[:,:-1].columns.values
        for col in columns:
            act24[col+'_bool']=''
            act24[col+'_duration']=''
            act24[col+'_bool'] = act24[col].ne(act24[col].shift(1))
            for i in range(len(act24)):
                s = act24.loc[act24.index[i],col]
                t = act24.loc[act24.index[i],col+'_bool']
                if s==1 and t==True:
                     act24.loc[act24.index[i],col+'_duration']=1
                elif s==1 and t==False:
                    act24.loc[act24.index[i],col+'_duration']=act24.loc[act24.index[i-1],col+'_duration']+1
                else:
                    act24.loc[act24.index[i],col+'_duration']=0
            act24.drop(columns = [col+'_bool'],axis=1, inplace=True)
            
        duration_columns = act24.columns.tolist()[-self.num_activities:]
        
        # Normalise price and emissions data
        data['price_norm'] = MinMaxScaler().fit_transform(np.array(data['price']).reshape(-1,1))
        data['avg_intensity_forecast_norm'] = MinMaxScaler().fit_transform(np.array(data['avg_intensity_forecast']).reshape(-1,1))

        em_savings_total1 = 0
        em_savings_total2 = 0
        em_savings_total3 = 0
        
        em_total1 = 0
        em_total2 = 0
        em_total3 = 0
        
        pr_savings_total1 = 0
        pr_savings_total2 = 0
        pr_savings_total3 = 0
        
        pr_total1 = 0
        pr_total2 = 0
        pr_total3 = 0
        
        rec_dict = collections.defaultdict(list)
        save_dict = collections.defaultdict(list)
        
        rec_cooking = collections.defaultdict(list)
        rec_working = collections.defaultdict(list)
        rec_entertainment = collections.defaultdict(list)
        rec_laundering = collections.defaultdict(list)
        rec_cleaning = collections.defaultdict(list)
        
        # Get dictionary in the form of d{predicted beginning hour of activity : duration of activity}
        for col in duration_columns:
            d = {}
            best = {}
            for i in range(len(act24)-1):
                if act24.loc[act24.index[i],col]!=0:
                    if act24.loc[act24.index[i],col]==1:
                        hour = act24.loc[act24.index[i],'Hour']
                        count = 1
                    else:
                        count = act24.loc[act24.index[i],col]
                if act24.loc[act24.index[i],col]!=0 and act24.loc[act24.index[i+1],col]==0:
                    d.update({hour:count})
     
            act = col[:-14]
        
            #if len(d)!=0:
            #    print('\033[1m\n'+act+'\033[0m\n')
            #    print(d)
            
            # Recommendation for flexible activities
            # aval_off == True: all hours of the day of recommendation are in the range of possible hours of beginnings
            # aval_off == False: only hours of predicted availability are in the range of possible hours of beginnings
            if act in ['cleaning','laundering']:
                if len(d)!=0 and aval_hours:
                    for i in range(len(d)):
                        
                        l={}
                        prl={}
                        eml={}
                        con = 0
                        count = 0

                        hour = list(d.keys())[i]
                        duration = list(d.values())[i]
                    
                        # Get consumption of all devices used for activity
                        if type(ident_dev_act[act])==list:
                            dev_num = len(ident_dev_act[act])
                            for v in range(dev_num):
                                dev = ident_dev_act[act][v]
                                col = ident_dev_act[act][v]+'_usage'
                                if dev_use[col][dev_use['Hour']==hour].item()==1:
                                    con = con + duration * self.dev_avg_con.get(ident_dev_act[act][v])
                                else:
                                    count+=1
                                    con = con + duration * self.dev_avg_con.get(ident_dev_act[act][v])
                            if count==dev_num:
                                con = con/dev_num
                            
                        else:
                            dev = ident_dev_act[act]
                            col = ident_dev_act[act]+'_usage'
                            con = duration * self.dev_avg_con.get(ident_dev_act[act])
                        
                        con = con/1000
                       
                        if self.aval_off:
                            
                            # range = possible hours of beginnings
                            for j in range(act24.index.hour[0],act24.index.hour[0]+24):
                                    
                                if j>23:
                                    j=j-24 
                                    
                                pr_sum = 0
                                em_sum = 0
                                pr_norm_sum = 0
                                em_norm_sum = 0
                                
                                # Loop over hours of activity set by duration 
                                for k in range(j,j+duration):
                                    if k>23:
                                        k=k-24
                                    if k<0:
                                        k=k+24 
                                    
                                    # Get price of hour k 
                                    pr = data['price'][data['Hour']==k].tolist()
                                    # Get price depending on energy consumption of activity
                                    pr_con = pr[0] * con
                                    pr_sum = pr_sum + pr_con
                                    
                                    # Get normalised price of hour k
                                    pr_norm = data['price_norm'][data['Hour']==k].item()
                                    pr_norm_sum = pr_norm_sum + pr_norm

                                    # Get emissions of hour k
                                    em = data['avg_intensity_forecast'][data['Hour']==k].item()
                                    # Get emissions depending on energy consumption of activity
                                    em_con = em * con
                                    em_sum = em_sum + em_con
                                    
                                    # Get normalised emissions of hour k
                                    em_norm = data['avg_intensity_forecast_norm'][data['Hour']==k].item()
                                    em_norm_sum = em_norm_sum + em_norm
                                
                                # Get total normalised emissions and energy costs for activity 
                                # depending on the starting hour and the emissions ratio
                                total = em_norm_sum * self.emissions_ratio + pr_norm_sum * (1-self.emissions_ratio)
                                
                                #l{hour of beginning : total of normalised emissions produced + normalised energy costs}
                                l.update({j:total})
                                
                                # Save total energy costs and emissions depending on consumption and starting hour
                                prl.update({j:pr_sum})
                                eml.update({j:em_sum})
                            
                            # Energy costs and emissions without shifting
                            pr_ws = prl.get(hour)
                            pr_total1 = pr_total1 + pr_ws
                            em_ws = eml.get(hour)
                            em_total1 = em_total1 + em_ws
                    
                        else:
                            # range = possible hours of beginnings = hours of availability
                            for j in aval_hours: 
                                
                                pr_sum = 0
                                em_sum = 0
                                pr_norm_sum = 0
                                em_norm_sum = 0
                                
                                # Loop over hours of activity set by duration 
                                for k in range(j,j+duration):
                                    if k>23:
                                        k=k-24
                                    if k<0:
                                        k=k+24 
                                    
                                    # Get price of hour k 
                                    pr = data['price'][data['Hour']==k].tolist()
                                    # Get price depending on energy consumption of activity
                                    pr_con = pr[0] * con
                                    pr_sum = pr_sum + pr_con
                                    
                                    # Get normalised price of hour k
                                    pr_norm = data['price_norm'][data['Hour']==k].item()
                                    pr_norm_sum = pr_norm_sum + pr_norm

                                    # Get emissions of hour k
                                    em = data['avg_intensity_forecast'][data['Hour']==k].item()
                                    # Get emissions depending on energy consumption of activity
                                    em_con = em * con
                                    em_sum = em_sum + em_con
                                    
                                    # Get normalised emissions of hour k
                                    em_norm = data['avg_intensity_forecast_norm'][data['Hour']==k].item()
                                    em_norm_sum = em_norm_sum + em_norm
                                
                                # Get total normalised emissions and energy costs for activity 
                                # depending on the starting hour and the emissions ratio
                                total = em_norm_sum * self.emissions_ratio + pr_norm_sum * (1-self.emissions_ratio)
                                
                                #l{hour of beginning : total of normalised emissions produced + normalised energy costs}
                                l.update({j:total})
                                
                                # Save total energy costs and emissions depending on consumption and starting hour
                                prl.update({j:pr_sum})
                                eml.update({j:em_sum})
                        
                            if hour not in aval_hours:
                                pr_ws = 0
                                em_ws = 0

                                # Loop over hours of activity set by duration 
                                for k in range(hour,hour+duration):
                                    if k>23:
                                        k=k-24
                                    if k<0:
                                        k=k+24
                                    
                                    # Calculate energy costs without shifting
                                    pr = data['price'][data['Hour']==k].item()
                                    pr_con = pr * con
                                    pr_ws = pr_ws + pr_con
                                    pr_total1 = pr_total1 + pr_ws
                                    
                                    # Calculate emissions without shifting
                                    em = data['avg_intensity_forecast'][data['Hour']==k].item()
                                    em_con = em * con
                                    em_ws = em_ws + em_con
                                    em_total1 = em_total1 + em_ws
                                    
                            else:
                                # Energy costs and emissions without shifting
                                pr_ws = prl.get(hour)
                                pr_total1 = pr_total1 + pr_ws
                                em_ws = eml.get(hour)
                                em_total1 = em_total1 + em_ws
                        
                        # Get starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                        lowest = min(l.items(), key=lambda x: x[1]) 
                       
                        # if hour with lowest value is already recommended, get hour with the next smallest value
                        while lowest[0] in list(best.keys()) and len(l)>1:
                            del l[lowest[0]]
                            lowest = min(l.items(), key=lambda x: x[1]) 
                        
                        # Save starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                        # together with the activity's duration
                        best.update({lowest[0]:duration})
                        rec_dict[lowest[0]].append(1)
                        
                        if act == 'cleaning':
                            rec_cleaning[lowest[0]].append(1)
                        else: 
                            rec_laundering[lowest[0]].append(1)
                        
                        # Get emissions for starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                        em_s =  eml.get(lowest[0])
                        # Compute emissions savings through shifting
                        s = em_ws-em_s
                        
                        # Compute total emissions savings through flexible activities
                        em_savings_total1 = em_savings_total1 + s
                        
                        # Get energy costs for starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                        pr_s =  prl.get(lowest[0])
                        # Compute costs savings through shifting
                        s = pr_ws-pr_s
                        
                        # Compute total costs savings through flexible activities
                        pr_savings_total1 = pr_savings_total1 + s
                        
                if best:
                    print('\033[1m\n'+act+'\033[0m\n')
                    for i in range(len(best)):
                        print('Beginning hour: {0} Duration: {1}'.format(list(best.keys())[i],list(best.values())[i]))

            # Recommendation for slightly flexible activities        
            if act in ['entertainment']:
                act_hours = list(d.keys())
                
                # Get intersection of predicted hours of availability and predicted activity starting hours
                intersec = [x for x in aval_hours if x in set(act_hours)]
                for hour in intersec:
                    l={}
                    eml={}
                    prl={}
                    con = 0
                    count = 0
                    
                    duration = d[hour]
                    
                    # Get consumption of all devices used for activity
                    if type(ident_dev_act[act])==list:
                        dev_num = len(ident_dev_act[act])
                        for v in range(dev_num):
                            dev = ident_dev_act[act][v]
                            col = ident_dev_act[act][v]+'_usage'
                            if dev_use[col][dev_use['Hour']==hour].item()==1:
                                con = con + duration * self.dev_avg_con.get(ident_dev_act[act][v])
                            else:
                                count+=1
                                con = con + duration * self.dev_avg_con.get(ident_dev_act[act][v])
                        if count==dev_num:
                            con = con/dev_num
                    else:
                        dev = ident_dev_act[act]
                        col = ident_dev_act[act]+'_usage'
                        #if dev_use[col][dev_use['Hour']==hour].item()==1:
                        con = duration * self.dev_avg_con.get(ident_dev_act[act])
                    con = con/1000
                    
                    # range = possible hours of beginnings
                    # for slightly flexible activities possible shift is set to 1 hour before and 4 hours after the predicted starting hour
                    for j in range(hour-1,hour+4): 
                        if j>23:
                            j=j-24
                        if j<0:
                            j=j+24
                            
                        em_sum = 0
                        pr_sum = 0
                        pr_norm_sum = 0
                        em_norm_sum = 0
                        
                        # Loop over hours of activity set by duration 
                        for k in range(j,j+d[hour]):
                            if k>23:
                                k=k-24
                            if k<0:
                                k=k+24 
                                    
                            # Get price of hour k 
                            pr = data['price'][data['Hour']==k].item()
                            # Get price depending on energy consumption of activity
                            pr_con = pr * con
                            pr_sum = pr_sum + pr_con
                                    
                            # Get normalised price of hour k
                            pr_norm = data['price_norm'][data['Hour']==k].item()
                            pr_norm_sum = pr_norm_sum + pr_norm

                            # Get emissions of hour k
                            em = data['avg_intensity_forecast'][data['Hour']==k].item()
                            # Get emissions depending on energy consumption of activity
                            em_con = em * con
                            em_sum = em_sum + em_con
                                    
                            # Get normalised emissions of hour k
                            em_norm = data['avg_intensity_forecast_norm'][data['Hour']==k].item()
                            em_norm_sum = em_norm_sum + em_norm
                                
                        # Get total normalised emissions and energy costs for activity 
                        # depending on the starting hour and the emissions ratio
                        total = em_norm_sum * self.emissions_ratio + pr_norm_sum * (1-self.emissions_ratio)
                                
                        #l{hour of beginning : total of normalised emissions produced + normalised energy costs}
                        l.update({j:total})
                                
                        # Save total energy costs and emissions depending on consumption and starting hour
                        prl.update({j:pr_sum})
                        eml.update({j:em_sum})
                    
                    # Get starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                    lowest = min(l.items(), key=lambda x: x[1])  
                    
                    # Save starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                    # together with the activity's duration
                    best.update({lowest[0]:duration})
                    rec_dict[lowest[0]].append(1)
                    rec_entertainment[lowest[0]].append(1)
                    
                    # Energy costs and emissions without shifting
                    pr_ws = prl.get(hour)
                    pr_total2 = pr_total2 + pr_ws
                    em_ws = eml.get(hour)
                    em_total2 = em_total2 + em_ws
                    
                    # Get emissions for starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                    em_s =  eml.get(lowest[0])
                    # Compute emissions savings through shifting
                    s = em_ws-em_s

                    # Compute total emissions savings through slightly flexible activities
                    em_savings_total2 = em_savings_total2 + s
                    
                    # Get energy costs for starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                    pr_s =  prl.get(lowest[0])
                    # Compute costs savings through shifting
                    s = pr_ws-pr_s
                    
                    # Compute total costs savings through slightly flexible activities
                    pr_savings_total2 = pr_savings_total2 + s
                    
                if best:
                    print('\033[1m\n'+act+'\033[0m\n')
                    for i in range(len(best)):
                        print('Beginning hour: {0} Duration: {1}'.format(list(best.keys())[i],list(best.values())[i]))
        
            if act in ['cooking','working']:
                act_hours = list(d.keys())
                
                # Get intersection of predicted hours of availability and predicted activity starting hours
                intersec = [x for x in aval_hours if x in set(act_hours)]

                for hour in intersec:
                    l={}
                    eml={}
                    prl={}
                    con = 0
                    count = 0
                    
                    duration = d[hour]
                    
                    # Get consumption of all devices used for activity
                    if type(ident_dev_act[act])==list:
                        dev_num = len(ident_dev_act[act])
                        for v in range(dev_num):
                            dev = ident_dev_act[act][v]
                            col = ident_dev_act[act][v]+'_usage'
                            if dev_use[col][dev_use['Hour']==hour].item()==1:
                                con = con + duration * self.dev_avg_con.get(ident_dev_act[act][v])
                            else:
                                count+=1
                                con = con + duration * self.dev_avg_con.get(ident_dev_act[act][v])
                        if count==dev_num:
                                con = con/dev_num
                    else:
                        dev = ident_dev_act[act]
                        col = ident_dev_act[act]+'_usage'
                        #if dev_use[col][dev_use['Hour']==hour].item()==1:
                        con = duration * dev_avg_con.get(ident_dev_act[act])
                    con = con/1000
                    
                    # range = possible hours of beginnings
                    # for inflexible activities possible shift is set to 1 hour before and 2 hours after the predicted starting hour
                    for j in range(hour-1,hour+2): 
                        if j>23:
                            j=j-24
                        if j<0:
                            j=j+24
                            
                        em_sum = 0
                        pr_sum = 0
                        pr_norm_sum = 0
                        em_norm_sum = 0
                        
                        # Loop over hours of activity set by duration 
                        for k in range(j,j+d[hour]):
                            if k>23:
                                k=k-24
                            if k<0:
                                k=k+24 
                                    
                            # Get price of hour k 
                            pr = data['price'][data['Hour']==k].item()
                            # Get price depending on energy consumption of activity
                            pr_con = pr * con
                            pr_sum = pr_sum + pr_con
                                    
                            # Get normalised price of hour k
                            pr_norm = data['price_norm'][data['Hour']==k].item()
                            pr_norm_sum = pr_norm_sum + pr_norm

                            # Get emissions of hour k
                            em = data['avg_intensity_forecast'][data['Hour']==k].item()
                            # Get emissions depending on energy consumption of activity
                            em_con = em * con
                            em_sum = em_sum + em_con
                                    
                            # Get normalised emissions of hour k
                            em_norm = data['avg_intensity_forecast_norm'][data['Hour']==k].item()
                            em_norm_sum = em_norm_sum + em_norm
                                
                        # Get total normalised emissions and energy costs for activity 
                        # depending on the starting hour and the emissions ratio
                        total = em_norm_sum * self.emissions_ratio + pr_norm_sum * (1-self.emissions_ratio)
                                
                        #l{hour of beginning : total of normalised emissions produced + normalised energy costs}
                        l.update({j:total})
                                
                        # Save total energy costs and emissions depending on consumption and starting hour
                        prl.update({j:pr_sum})
                        eml.update({j:em_sum})
                    
                    # Get starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                    lowest = min(l.items(), key=lambda x: x[1]) 
                 
                    # Save starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                    # together with the activity's duration
                    best.update({lowest[0]:duration})
                    rec_dict[lowest[0]].append(1)
                    
                    if act == 'cooking':
                        rec_cooking[lowest[0]].append(1)
                    else: 
                        rec_working[lowest[0]].append(1)
                        
                    # Energy costs and emissions without shifting
                    pr_ws = prl.get(hour)
                    pr_total3 = pr_total3 + pr_ws
                    em_ws = eml.get(hour)
                    em_total3 = em_total3 + em_ws

                    # Get emissions for starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                    em_s =  eml.get(lowest[0])
                    # Compute emissions savings through shifting
                    s = em_ws-em_s
                    
                    # Compute total emissions savings through inflexible activities
                    em_savings_total3 = em_savings_total3 + s
                    
                    # Get energy costs for starting hour with minimum normalised emissions and energy costs depending on emissions ratio
                    pr_s =  prl.get(lowest[0])
                    # Compute costs savings through shifting
                    s = pr_ws-pr_s
                    
                    # Compute total costs savings through inflexible activities
                    pr_savings_total3 = pr_savings_total3 + s
                
                if best:
                    print('\033[1m\n'+act+'\033[0m\n')
                    for i in range(len(best)):
                        print('Beginning hour: {0} Duration: {1}'.format(list(best.keys())[i],list(best.values())[i]))
        
        # Get total emissions and energy costs without shifting
        em_total = sum((em_total1,em_total2,em_total3))
        pr_total = sum((pr_total1,pr_total2,pr_total3))
        
        em_total = round(em_total,0)
        pr_total = round(pr_total,2)
        
        save_dict['em_total'].append(em_total)
        save_dict['pr_total'].append(pr_total)
        
        # Get total emissions and energy costs savings through activity shifting
        em_savings_total = sum((em_savings_total1,em_savings_total2,em_savings_total3))
        pr_savings_total = sum((pr_savings_total1,pr_savings_total2,pr_savings_total3))
        
        em_savings_total = round(em_savings_total,0)
        pr_savings_total = round(pr_savings_total,2)
        
        save_dict['em_savings'].append(em_savings_total)
        save_dict['pr_savings'].append(pr_savings_total)
        
        print('\n\033[1mTotal emissions savings from shifting: {0} gCO2\033[0m\n'.format('{:.0f}'.format(em_savings_total)))
        print('\n\033[1mTotal price savings from shifting: {0} €\033[0m\n'.format('{:.2f}'.format(pr_savings_total)))
    
        return rec_dict, save_dict, rec_cooking, rec_working, rec_entertainment, rec_laundering, rec_cleaning 
