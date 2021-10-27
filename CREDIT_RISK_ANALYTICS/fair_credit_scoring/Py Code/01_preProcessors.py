# -*- coding: utf-8 -*-
"""
PREPROCESSING
Created on Mon Feb  3 14:47:12 2020

The custom pre-processing function is adapted from
https://github.com/IBM/AIF360

@author: Johannes
"""
# output_path = 'C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\3_wipResults\\'
# Load all necessary packages
import sys
sys.path.append("../")
# sys.path.append("C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\fairCreditScoring\\py_code\\1_PRE\\1_code")
import numpy as np

#from load_germandata import load_GermanDataset
from load_taiwandata import load_TaiwanDataset
#from load_gmscdata import load_GMSCDataset

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing.lfr import LFR
from aif360.algorithms.preprocessing import DisparateImpactRemover

from sklearn.preprocessing import MaxAbsScaler

#import matplotlib.pyplot as plt


## import datasets
dataset_orig = load_TaiwanDataset() #load_GMSCDataset() 

protected = 'AGE'
privileged_groups = [{'AGE': 1}] 
unprivileged_groups = [{'AGE': 0}]
print(dataset_orig.feature_names)


#random seed for calibrated equal odds prediction
np.random.seed(1)

# Get the dataset and split into train and test
dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
# =============================================================================
tr, val, te = dataset_orig_train.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True), \
              dataset_orig_valid.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True), \
              dataset_orig_test.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)
 
tr[0].to_csv(output_path + 'taiwan_' + 'orig_train' + '.csv', index = None, header=True)
te[0].to_csv(output_path + 'taiwan_' + 'orig_test' + '.csv', index = None, header=True)
val[0].to_csv(output_path + 'taiwan_' + 'orig_valid' + '.csv', index = None, header=True)
# =============================================================================

# Scale data and check that the Difference in mean outcomes didn't change
min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
dataset_orig_valid.features = min_max_scaler.transform(dataset_orig_valid.features)

# =============================================================================
tr, val, te = dataset_orig_train.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True), \
              dataset_orig_valid.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True), \
              dataset_orig_test.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)
 
tr[0].to_csv(output_path + 'taiwan_' + 'scaled_train' + '.csv', index = None, header=True)
te[0].to_csv(output_path + 'taiwan_' + 'scaled_test' + '.csv', index = None, header=True)
val[0].to_csv(output_path + 'taiwan_' + 'scaled_valid' + '.csv', index = None, header=True)
# =============================================================================
              
# Preprocessing
methods = ["reweighing"#, "disp_impact_remover"
           ]

for m in methods:
    if m == "reweighing":
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                       privileged_groups=privileged_groups)
        RW.fit(dataset_orig_train)
        
        # train classification
        dataset_transf_train = RW.transform(dataset_orig_train)
        w_train = dataset_transf_train.instance_weights.ravel()
        out_train = dataset_transf_train.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)[0]
        out_train = out_train.sample(n=out_train.shape[0], replace=True, weights=w_train)
        
        # valid classification
        dataset_transf_valid = RW.transform(dataset_orig_valid)
        w_valid = dataset_transf_valid.instance_weights.ravel()
        out_valid = dataset_transf_valid.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)[0]
        out_valid = out_valid.sample(n=out_valid.shape[0], replace=True, weights=w_valid)
        
        # test classification
        dataset_transf_test = RW.transform(dataset_orig_test)
        w_test = dataset_transf_test.instance_weights.ravel()
        out_test = dataset_transf_test.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)[0]
        out_test = out_test.sample(n=out_test.shape[0], replace=True, weights=w_test)
        
        
        # Code testing that transformation worked
        assert np.abs(dataset_transf_train.instance_weights.sum()-dataset_orig_train.instance_weights.sum())<1e-6

    elif m == "lfr":
        TR = LFR(unprivileged_groups = unprivileged_groups, privileged_groups = privileged_groups)
        TR = TR.fit(dataset_orig_train)
        dataset_transf_train = TR.transform(dataset_orig_train, threshold = 0.8)
        out = dataset_transf_train.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)[0]
       
    elif m == "disp_impact_remover":     
        # Test if scaling changes something --> but then also export a scaled test set
        # scaler = MinMaxScaler(copy=False)
        
        di = DisparateImpactRemover(repair_level=1, sensitive_attribute='AGE')
        dataset_transf_train = di.fit_transform(dataset_orig_train)
        out_train = dataset_transf_train.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)[0]

        # valid classification
        dataset_transf_valid = di.fit_transform(dataset_orig_valid)
        out_valid = dataset_transf_valid.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)[0]
        
        # test classification
        dataset_transf_test = di.fit_transform(dataset_orig_test)
        out_test = dataset_transf_test.convert_to_dataframe(de_dummy_code=True, sep='=', set_category=True)[0]
        

    metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    print(m + " achieved a statistical parity difference between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

        
    out_train.to_csv(output_path + 'taiwan_pre_train_' + m + '.csv', index = None, header=True)
    out_valid.to_csv(output_path + 'taiwan_pre_valid_' + m + '.csv', index = None, header=True)
    out_test.to_csv(output_path + 'taiwan_pre_test_' + m + '.csv', index = None, header=True)









