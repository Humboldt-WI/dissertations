import numpy as np
import torch
import pandas as pd

def pred_acc(pred, price_before, price_actual, pred_type):
    if pred_type == 'price':
        predictions = (pred - price_before) / price_before
        true = (price_actual - price_before) / price_before
    if pred_type == 'ret':
        predictions = pred.copy()
        true = (price_actual - price_before) / price_before
    if pred_type == 'binclass':
        predictions = pred.copy()
        true = (price_actual - price_before) / price_before

    a, d_stock, b = predictions.shape
    if a != 1 or b != 1: raise RuntimeError("Only one day prediction cycles. evaluation input expects [1,d_stock,1] but got ",predictions.shape,".")

    #account for non-trading days
    predictions_median = np.median(predictions)
    if pred_type != 'binclass':
        for i in range(0, len(predictions)):
            for j in range(0, len(predictions[0])):
                if true[i, j] == 0:
                    predictions[i, j] = predictions_median + np.finfo(float).eps
                j = j + 1
            i = i + 1

    #classify predictions
    if pred_type != 'binclass':
        class_predictions = np.zeros_like(predictions)
        class_predictions[predictions < predictions_median] = 0
        class_predictions[predictions >= predictions_median] = 1
    else:
        class_predictions = predictions
        #following three added for binclass transformer. Possibl. buggy for LSTM.
        class_predictions = np.zeros_like(predictions)
        class_predictions[predictions < predictions_median] = 0
        class_predictions[predictions >= predictions_median] = 1

    #classify true
    true_median = np.median(true)
    class_true = np.zeros_like(true)
    class_true[true < true_median] = 0
    class_true[true >= true_median] = 1

    #true-false table
    tab = np.array(pd.crosstab(torch.flatten(torch.Tensor(class_predictions)), torch.flatten(torch.Tensor(class_true))))
    tab = tab / sum(sum(tab))

    p_dim, t_dim = tab.shape

    if p_dim == 2 and t_dim == 2:
        accuracy_stats = np.array([tab[1,1] + tab[0,0], tab[1,1], tab[0,0], tab[1,0], tab[0,1]])
        #ordering: acc., pTop_tTop, pFlop_tFlop, pTop_tFlop, pFlop_tTop
    elif p_dim == 2:
        if class_true[0,0,0] == 0: #all true labels are flop
            accuracy_stats = np.array([tab[0, 0], 0, tab[0, 0], tab[1, 0], 0])
        elif class_true[0,0,0] == 1: #all true labels are top
            accuracy_stats = np.array([tab[1, 0], tab[1, 0], 0, 0, tab[0, 0]])
        else:
            accuracy_stats = np.array([0, 0, 0, 0, 0])
    elif t_dim == 2:
        if class_predictions[0,0,0] == 0: #all prediction labels are flop
            accuracy_stats = np.array([tab[0, 0], 0, tab[0, 0], 0, tab[0, 1]])
        elif class_predictions[0,0,0] == 1: #all prediction lables are top
            accuracy_stats = np.array([tab[0, 1], tab[0, 1], 0, tab[0, 0], 0])
        else:
            accuracy_stats = np.array([0, 0, 0, 0, 0])
    elif t_dim == p_dim == 1:
        if class_predictions[0,0,0] == class_true[0,0,0] == 0: #all labels flop
            accuracy_stats = np.array([1, 0, 1, 0, 0])
        elif class_predictions[0,0,0] == class_true[0,0,0] == 1: #all labels top
            accuracy_stats = np.array([1, 1, 0, 0, 0])
        else:
            accuracy_stats = np.array([0, 0, 0, 0, 0])
    else:
        accuracy_stats = np.array([0, 0, 0, 0, 0])
    return accuracy_stats

