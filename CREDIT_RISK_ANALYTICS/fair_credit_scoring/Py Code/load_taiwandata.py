# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:32:53 2020

LOADING GERMAN CREDIT DATASET

@author: Johannes
"""

import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StandardDataset
import numpy as np

def load_TaiwanDataset():
    
    filepath = "C:\\Users\\Johannes\\Desktop\\Code - Copy\\data\\UCI_Credit_Card.csv"
    df = pd.read_csv(filepath, sep=',', na_values=[])
    
    df = df.rename(columns={'default.payment.next.month': 'TARGET'})
    del df['ID']
    df['AGE'] = df['AGE'].apply(lambda x: np.where(x > 25, 1.0, 0.0))
    df['CREDIT_AMNT'] = df['BILL_AMT1'] - df['PAY_AMT1']

    XD_features = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE","PAY_0",
                "PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","BILL_AMT1",
                "BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
                "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5",
                "PAY_AMT6", "CREDIT_AMNT"]
    D_features = ['AGE']
    Y_features = ['TARGET']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', "PAY_0",
                            "PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
    
    privileged_class = {"AGE": [1.0]}
    protected_attribute_map = {"AGE": {1.0: 'Old', 0.0: 'Young'}}
    
  
    def default_preprocessing(df):
        
        def label_sex(x):
            if x == 1:
                return 'Male'
            elif x == 2:
                return 'Female'
            else:
                return 'NA'

        def label_education(x):
            if x == 1:
                return 'graduate_school'
            elif x == 2:
                return 'university'
            elif x == 3:
                return 'high_school'
            elif x == 4:
                return 'others'
            elif x == 5:
                return 'others'
            elif x == 6:
                return 'others'
            else:
                return 'others'

        def label_marriage(x):
            if x == 1:
                return 'married'
            elif x == 2:
                return 'single'
            elif x == 3:
                return 'others'
            else:
                return 'others'
        
        #to be defined
        def label_pay(x):
            if x in [-2,-1]:
                return 0
            else:
                return x
    
        # group credit history, savings, and employment
        df['SEX'] = df['SEX'].apply(lambda x: label_sex(x))
        df['EDUCATION'] = df['EDUCATION'].apply(lambda x: label_education(x))
        df['MARRIAGE'] = df['MARRIAGE'].apply(lambda x: label_marriage(x))
        
        pay_col = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
        for p in pay_col:
            df[p] = df[p].apply(lambda x: label_pay(x))
        
        # Good credit == 1
        status_map = {0: 1.0, 1: 2.0}
        df['TARGET'] = df['TARGET'].replace(status_map)

                
        return df
    
    df_standard = StandardDataset(
        df = df,
        label_name=Y_features[0],
        favorable_classes=[1],
        protected_attribute_names=D_features,
        privileged_classes=[privileged_class["AGE"]],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        metadata={'label_maps': [{1.0: 'Good', 2.0: 'Bad'}],
                   'protected_attribute_maps': [protected_attribute_map]},
        custom_preprocessing=default_preprocessing)
    
    return df_standard
