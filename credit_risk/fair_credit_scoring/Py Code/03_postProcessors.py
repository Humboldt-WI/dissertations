# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:19:48 2020

@author: Johannes
"""

#output_path = 'C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\3_wipResults\\'
# Load all necessary packages
import sys
sys.path.append("../")
#sys.path.append("C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\fairCreditScoring\\py_code\\1_PRE\\1_code")
import numpy as np
import pandas as pd

#from load_germandata import load_GermanDataset
from load_taiwandata import load_TaiwanDataset
#from load_gmscdata import load_GMSCDataset

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.postprocessing.reject_option_classification\
        import RejectOptionClassification
from aif360.algorithms.postprocessing.eq_odds_postprocessing\
        import EqOddsPostprocessing

from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression

## import prediction results from R BASED ON THE SAME PREDICTIONS FOR COMPARISON
dataset_trainResults_valid = pd.read_csv(output_path + 'taiwan_post_training_results_dval.csv')
dataset_trainResults_test = pd.read_csv(output_path + 'taiwan_post_training_results_dtest.csv')

## import dataset
dataset_orig = load_TaiwanDataset() 

# Scale all vars: di_remover = minmaxscaling, rest = standard_scaling
protected = 'AGE'
privileged_groups = [{'AGE': 1}] 
unprivileged_groups = [{'AGE': 0}]
print(dataset_orig.feature_names)

# Metric used (should be one of allowed_metrics)
metric_name = "Statistical parity difference"

# Upper and lower bound on the fairness metric used
metric_ub = 0.05
metric_lb = -0.05

#random seed for calibrated equal odds prediction
np.random.seed(1)

# Get the dataset and split into train and test
dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.7], shuffle=True)
dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)

# Scale data and check that the Difference in mean outcomes didn't change
min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
dataset_orig_valid.features = min_max_scaler.transform(dataset_orig_valid.features)

dataset_orig_valid_pred = dataset_orig_valid.copy(deepcopy=True)
dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)

# Postprocessing
model_names = ['glm', "svmLinear", "rf", "xgbTree", "nnet"]
ROC_test = pd.DataFrame()

for m in model_names:
    scores_valid = np.array(dataset_trainResults_valid[m+'_scores']).reshape(len(dataset_trainResults_valid.index),1)
    labels_valid = np.where(dataset_trainResults_valid[m+'_class']=='Good', 1.0, 2.0).reshape(len(dataset_trainResults_valid.index),1)
    
    scores_test = np.array(dataset_trainResults_test[m+'_scores']).reshape(len(dataset_trainResults_test.index),1)
    labels_test = np.where(dataset_trainResults_test[m+'_class']=='Good', 1.0, 2.0).reshape(len(dataset_trainResults_test.index),1)

    dataset_orig_valid_pred.scores = scores_valid
    dataset_orig_valid_pred.labels = labels_valid
    
    dataset_orig_test_pred.scores = scores_test
    dataset_orig_test_pred.labels = labels_test
    
    # Reject Option Classification
    ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, 
                                     privileged_groups=privileged_groups, 
                                     low_class_thresh=0.01, high_class_thresh=0.99,
                                      num_class_thresh=100, num_ROC_margin=50,
                                      metric_name=metric_name,
                                      metric_ub=metric_ub, metric_lb=metric_lb)
    ROC = ROC.fit(dataset_orig_valid, dataset_orig_valid_pred)
      
    # ROC_test results
    dataset_transf_test_pred = ROC.predict(dataset_orig_test_pred)
    
    ROC_test[m+"_fairScores"] = dataset_transf_test_pred.scores.flatten()
    label_names = np.where(dataset_transf_test_pred.labels==1,'Good','Bad')
    ROC_test[m+"_fairLabels"] = label_names
    
ROC_test.to_csv(output_path + 'taiwan_post_roc_results_test.csv', index = None, header=True)







