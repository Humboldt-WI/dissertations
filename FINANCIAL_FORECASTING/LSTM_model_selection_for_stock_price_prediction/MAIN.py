# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:32:44 2021

@author: Valentin Millik
"""
# pylint: disable=invalid-name
import time
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import uniform, randint
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from bayes_opt import BayesianOptimization
import tensorflow as tf
from RNN_Functions import lstm

# Set Random Seeds
np.random.seed(1)
tf.random.set_seed(2)

# Constants
optM = "BO"
PLOT = False
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 1
OUT_OF_SAMPLE_PCT = 0.2
column_names = ['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']

# read csv and drop unnecessary columns.
df = pd.read_csv('IBM_Adj-Cl-H-L-V.csv', header=None, names=column_names,
                 index_col='Date')
df.drop(columns=['High', 'Low', 'Open', 'Close'], inplace=True)
df.index = pd.to_datetime(df.index, infer_datetime_format=True)


def classify(current, future):
    """
    Classifies smaples: if future price is bigger than current price -> 1, else: 0
    """
    if float(future) > current: # CHECK: If classification is right.
        a = 1
    else:
        a = 0
    return a

def preprocess(dataf):
    """
    Preprocesses data...
    """
    dataf = dataf.drop(columns=f'ShiftedPrice_by_{FUTURE_PERIOD_PREDICT}_days')
    
    # Calculate returns for Adj Closing price column and standardize features
    for col in dataf.columns:
        if col != "Target":
            if col != "Volume":
                dataf[col] = dataf[col].pct_change()
            dataf[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            dataf[col] = preprocessing.scale(dataf[col].to_numpy())
    dataf.dropna(inplace=True)

    # Creating Sequences
    sequential_data = []
    prevs_days = deque(maxlen=SEQ_LEN)

    for i in dataf.to_numpy():
        features = []
        for n in i[:-1]:
            features.append(n)
        prevs_days.append(features)
        if len(prevs_days) == SEQ_LEN:
            sequential_data.append([list(prevs_days), i[-1]])
    
    # Handling imbalanced data
    buys = []
    sells = []
    counter_S = 0
    counter_B = 0

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
            counter_S += 1
        elif target == 1:
            buys.append([seq, target])
            counter_B += 1

    print(f"Amount Buys: {counter_B}")
    print(f"Amount Sells: {counter_S}")

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells
    random.shuffle(sequential_data)

    # Count ocurrences of classes in sequences
    counter_S = 0
    counter_B = 0

    for seq, target in sequential_data:
        if target == 0:
            counter_S += 1
        elif target == 1:
            counter_B += 1

    print(f"Amount Buys: {counter_B}")
    print(f"Amount Sells: {counter_S}")


    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    X = np.asarray(X)
    y = np.asarray(y, dtype=np.int8)

    return X, y

def create_function(epochs, batch_size, n_1, dr_1, lr):
    """
    Function which creates lstm and returns performance measure
    for bayesian optimization
    @params:
        epochs      -   Required: Amount of epochs to train the LSTM.
        batch_size  -   Required: Amount of samples that are
                                  transferred to LSTM.
        n_1         -   Required: Amount Neurons of first Layer.
        dr_1        -   Required: Dropout Rate of first Layer.
        lr          -   Required: Learning Rate of LSTM.
    """
    # create Keras Wrapper for model
    model = KerasClassifier(build_fn=lstm,
                            X_shape=X_train.shape[1:],
                            lr=lr,
                            n_1=int(n_1),
                            dr_1=dr_1,
                            batch_size=int(batch_size),
                            epochs=int(epochs),
                            verbose=0)

    # use StratifiedKFold and apply cross_val_score
    results = cross_val_score(estimator=model, X=X_train, y=y_train, scoring='accuracy', cv=kfold, n_jobs=-1, verbose=1)
    return results.mean()


# Shifting Price
df[f'ShiftedPrice_by_{FUTURE_PERIOD_PREDICT}_days'] = \
    df['Adj Close'].shift(-FUTURE_PERIOD_PREDICT)

# Drop NaN's
df.dropna(inplace=True)

# Creating targets for learning
print(f"Amount Samples: {df['Adj Close'].shape[0]}")
df['Target'] = list(map(classify, df['Adj Close'],
                        df[f'ShiftedPrice_by_{FUTURE_PERIOD_PREDICT}_days']))

# Creating out-of-sample 'test'-set
times = sorted(df.index.values)
oos_test_set_dates = times[-int(OUT_OF_SAMPLE_PCT*len(times))]

test_data = df[(df.index >= oos_test_set_dates)]
training_data = df[(df.index < oos_test_set_dates)]
print(f"Amount Samples Training Set: {training_data['Adj Close'].shape[0]}")

# Preprocessing data
X_train, y_train = preprocess(training_data) # TODO: Value Counts of Target Var
print(f"X_train Shape: {X_train.shape}")
X_test, y_test = preprocess(test_data)

if PLOT:
    # Plotting for Visualization
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(training_data.index.to_numpy(), training_data.iloc[:, 1].to_numpy(),
 		    color='dodgerblue', label="Training Set")
    ax.plot(test_data.index.to_numpy(), 
		    test_data.iloc[:, 1].to_numpy(), color='orange',
		    label="Test Set")
    ax.set_xlabel("Datum", fontdict={'fontsize': 12}, labelpad=10)
    ax.set_ylabel("Preis (in USD)", fontdict={'fontsize': 12}, labelpad=10)
    plt.title("IBM Aktienkurs", fontdict={'fontsize': 18}, pad=20)
    plt.legend()
    plt.show()

# Grid Search Parameters
learningRate = list(np.linspace(0.00001, 0.01, 3, dtype=float))
amountNeurons = list(np.linspace(25, 75, 3, dtype=int))
dropoutRate = list(np.linspace(0.2, 0.8, 3, dtype=float))
EPOCHS = list(np.linspace(25, 75, 3, dtype=int))
BATCH_SIZE = list(np.linspace(32, 128, 3, dtype=int))

# Parameter Dictionary for Grid Search
param_grid = dict(lr=learningRate, n_1=amountNeurons, 
                  dr_1=dropoutRate,
                  batch_size=BATCH_SIZE, 
                  epochs=EPOCHS)

# Parameter Dictionary for Random Search
param_grid_RS = dict(lr=uniform(loc=0.00001, scale=0.00999),
                     n_1=randint(low=25, high=76),
                     dr_1=uniform(loc=0.2, scale=0.8),
                     batch_size=randint(low=32, high=128),
                     epochs=randint(low=25, high=75))

# Cross Validation StratifiedKFold
kfold = StratifiedKFold(n_splits=3, shuffle=True)


if optM == "GS":
    # Build model with sci-kit wrapper
    model = KerasClassifier(build_fn=lstm,
                            X_shape=X_train.shape[1:],
                            verbose=0)

    opt = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', n_jobs=-1, cv=kfold, verbose=2)
    results = opt.fit(X_train, y_train)
    results_df = pd.DataFrame(results.cv_results_)
    results_df.to_csv("GS_CV_results.csv", index=False)

    print(f"Best: {results.best_score_} using {results.best_params_}")
elif optM == "RS":
    model = KerasClassifier(build_fn=lstm,
                            X_shape=X_train.shape[1:],
                            verbose=0)

    opt = RandomizedSearchCV(estimator=model, param_distributions=param_grid_RS, n_iter=243, scoring='accuracy',
                               n_jobs=-1, cv=kfold, random_state=1, verbose=2)
    results = opt.fit(X_train, y_train)
    results_df = pd.DataFrame(results.cv_results_)
    results_df.to_csv("RS_BO_results.csv", index=False)

    print(f"Best: {results.best_score_} using {results.best_params_}")
elif optM == "BO":
    # Bounds of Hyperparamters to tune.
    pbounds = {'epochs': (25.0, 75.0), 'batch_size': (32.0, 128.0), 'n_1': (25.0, 75.0),
               'dr_1': (0.2, 0.8), 'lr': (0.00001, 0.01)}

    # create optimizer for Bayesian Opimization.
    optimizer = BayesianOptimization(f=create_function,
                                     pbounds=pbounds,
                                     verbose=3,
                                     random_state=1)

    optimizer.maximize(init_points=2,
                       n_iter=243)

    # Get best Parameter Kombination and respective Target Value on Validation Set
    print(optimizer.max)

    # Get History of Parameters and Target Values
    results = optimizer.res

    # Format the Parameter List for later plotting.
    results_list = []
    results_dict = {}
    for res in results:
        id = 0
        for k, v in res.items():
            if id == 0:
                target = v
            elif id == 1:
                results_dict = v
                results_dict['target'] = target
            else:
                print("Es ist ein Fehler beim Auslesen der Ergebnisse BO aufgetreten.")
            id += 1
        results_list.append(results_dict)


    results_df = pd.DataFrame(results_list)
    results_df.to_csv("BO_CV_results.csv")
elif optM == "NO":
    print("No opt. applied!")
    # Test model wrapper
    model = KerasClassifier(build_fn=lstm,
                            X_shape=X_train.shape[1:],
                            lr=0.001,
                            n_1=25,
                            dr_1=0.2,
                            epochs=50,
                            batch_size=60,
                            verbose=2)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(model.get_params())
    print(f"Final Accuracy: {accuracy}")
else:
    print("Falsche Eingabe!")


# Final Evaluation on Test Set
if optM == "GS" or optM == "RS":
    predictions = opt.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Final Accuracy on Test Set: {accuracy}")
