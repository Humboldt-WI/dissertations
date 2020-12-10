# -*- coding: utf-8 -*-
"""
INPROCESSING

Created on Mon Feb 17 10:08:22 2020

The custom pre-processing function is adapted from
https://github.com/IBM/AIF360

@author: Johannes
"""

#output_path = 'C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\3_wipResults\\'

# Load all necessary packages
import sys
sys.path.append("../")
#sys.path.append("C:\\Users\\Johannes\\OneDrive\\Dokumente\\Humboldt-Universität\\Msc WI\\1_4. Sem\\Master Thesis II\\fairCreditScoring\\py_code\\1_PRE\\1_code")
import numpy as np
import pandas as pd

from load_taiwandata import load_TaiwanDataset

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import tensorflow as tf

# Get the dataset and split into train and test
dataset_orig = load_TaiwanDataset() 

# Scale all vars: di_remover = minmaxscaling, rest = standard_scaling
protected = 'AGE'
privileged_groups = [{'AGE': 1}]
unprivileged_groups = [{'AGE': 0}]
print(dataset_orig.feature_names)

all_metrics =  ["Statistical parity difference",
                   "Average odds difference",
                   "Equal opportunity difference"]

#random seed for calibrated equal odds prediction
np.random.seed(1)

# Get the dataset and split into train and test
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True) #should be stratified for target

# Scaled dataset - Verify that the scaling does not affect the group label statistics¶
min_max_scaler = MaxAbsScaler()
dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

# Learn parameters with debiasing = with debias set to True
sess = tf.Session()
debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='debiased_classifier',
                          debias=True,
                          sess=sess)
debiased_model.fit(dataset_orig_train)

# Apply the plain model to test data
dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
dataset_debiasing_test = debiased_model.predict(dataset_orig_test)

scores = dataset_debiasing_test.scores

advdebias_predictions = pd.DataFrame()

advdebias_predictions["scores"] = sum(scores.tolist(), [])
    
advdebias_predictions.to_csv(output_path + 'taiwan_advdebias_predictions' + '.csv', index = None, header=True)

