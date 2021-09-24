import numpy as np
import torch

def get_trades(predictions, trueReturn, type):
    n_stocks = predictions.shape[1]

    predictions = torch.Tensor(predictions).reshape((len(predictions),len(predictions[0]),1)).clone()
    true = torch.Tensor(trueReturn).clone().detach()
    '''since the trading algorithm is based on trading the top/flop k stocks, one can effectively prevent it from
     choosing untraded stocks by setting predictions to median for the stocks on their non-trading days (those where
     trueReturn == 0). This will not eliminate all trading completely in those instances, where there are so few traded 
     stocks, that median-prediction stocks are still choosen. Equally, zero-return traded stocks may be wrongfully 
     interpreted as non-traded. Nevertheless, it is a closer approximation.'''

    if type != 'binclass':
        median = torch.median(predictions, dim=1)[0]
        for i in range(0,len(predictions)):
            for j in range(0,len(predictions[0])):
                if true[i,j] == 0:
                    predictions[i,j] = median[i] + np.finfo(float).eps
                j = j + 1
            i = i + 1

    if n_stocks > 5:
        trades_kV = np.zeros((len(predictions), len(predictions[0])))
        best_kV = torch.topk(predictions, 3, dim=1, largest=True, sorted=False).indices
        worst_kV = torch.topk(predictions, 3, dim=1, largest=False, sorted=False).indices
        trades_k1 = np.zeros((len(predictions), len(predictions[0])))
        best_k1 = torch.topk(predictions, 1, dim=1, largest=True, sorted=False).indices
        worst_k1 = torch.topk(predictions, 1, dim=1, largest=False, sorted=False).indices
    if n_stocks > 30:
        trades_k5 = np.zeros((len(predictions), len(predictions[0])))
        best_k5 = torch.topk(predictions, 5, dim=1, largest=True, sorted=False).indices
        worst_k5 = torch.topk(predictions, 5, dim=1, largest=False, sorted=False).indices
        trades_k10 = np.zeros((len(predictions), len(predictions[0])))
        best_k10 = torch.topk(predictions, 10, dim=1, largest=True, sorted=False).indices
        worst_k10 = torch.topk(predictions, 10, dim=1, largest=False, sorted=False).indices
        trades_k15 = np.zeros((len(predictions), len(predictions[0])))
        best_k15 = torch.topk(predictions, 15, dim=1, largest=True, sorted=False).indices
        worst_k15 = torch.topk(predictions, 15, dim=1, largest=False, sorted=False).indices
    if n_stocks > 50:
        trades_k20 = np.zeros((len(predictions), len(predictions[0])))
        best_k20 = torch.topk(predictions, 20, dim=1, largest=True, sorted=False).indices
        worst_k20 = torch.topk(predictions, 20, dim=1, largest=False, sorted=False).indices
        trades_k25 = np.zeros((len(predictions), len(predictions[0])))
        best_k25 = torch.topk(predictions, 25, dim=1, largest=True, sorted=False).indices
        worst_k25 = torch.topk(predictions, 25, dim=1, largest=False, sorted=False).indices
    if n_stocks > 100:
        trades_k30 = np.zeros((len(predictions), len(predictions[0])))
        best_k30 = torch.topk(predictions, 30, dim=1, largest=True, sorted=False).indices
        worst_k30 = torch.topk(predictions, 30, dim=1, largest=False, sorted=False).indices
        trades_k35 = np.zeros((len(predictions), len(predictions[0])))
        best_k35 = torch.topk(predictions, 35, dim=1, largest=True, sorted=False).indices
        worst_k35 = torch.topk(predictions, 35, dim=1, largest=False, sorted=False).indices
        trades_k40 = np.zeros((len(predictions), len(predictions[0])))
        best_k40 = torch.topk(predictions, 40, dim=1, largest=True, sorted=False).indices
        worst_k40 = torch.topk(predictions, 40, dim=1, largest=False, sorted=False).indices
        trades_k45 = np.zeros((len(predictions), len(predictions[0])))
        best_k45 = torch.topk(predictions, 45, dim=1, largest=True, sorted=False).indices
        worst_k45 = torch.topk(predictions, 45, dim=1, largest=False, sorted=False).indices
        trades_k50 = np.zeros((len(predictions), len(predictions[0])))
        best_k50 = torch.topk(predictions, 50, dim=1, largest=True, sorted=False).indices
        worst_k50 = torch.topk(predictions, 50, dim=1, largest=False, sorted=False).indices
    if n_stocks > 150:
        trades_k75 = np.zeros((len(predictions), len(predictions[0])))
        best_k75 = torch.topk(predictions, 75, dim=1, largest=True, sorted=False).indices
        worst_k75 = torch.topk(predictions, 75, dim=1, largest=False, sorted=False).indices
    if n_stocks > 200:
        trades_k100 = np.zeros((len(predictions), len(predictions[0])))
        best_k100 = torch.topk(predictions, 100, dim=1, largest=True, sorted=False).indices
        worst_k100 = torch.topk(predictions, 100, dim=1, largest=False, sorted=False).indices

    for i in range(0, len(predictions)): #i is the day
        if n_stocks > 5:
            for b in best_kV[i,:]:
                trades_kV[i][b] = 1
            for w in worst_kV[i,:]:
                trades_kV[i][w] = -1
            for b in best_k1[i,:]:
                trades_k1[i][b] = 1
            for w in worst_k1[i,:]:
                trades_k1[i][w] = -1
        if n_stocks > 30:
            for b in best_k5[i,:]:
                trades_k5[i][b] = 1
            for w in worst_k5[i,:]:
                trades_k5[i][w] = -1
            for b in best_k10[i,:]:
                trades_k10[i][b] = 1
            for w in worst_k10[i,:]:
                trades_k10[i][w] = -1
            for b in best_k15[i,:]:
                trades_k15[i][b] = 1
            for w in worst_k15[i,:]:
                trades_k15[i][w] = -1
        if n_stocks > 50:
            for b in best_k20[i,:]:
                trades_k20[i][b] = 1
            for w in worst_k20[i,:]:
                trades_k20[i][w] = -1
            for b in best_k25[i,:]:
                trades_k25[i][b] = 1
            for w in worst_k25[i,:]:
                trades_k25[i][w] = -1
        if n_stocks > 100:
            for b in best_k30[i,:]:
                trades_k30[i][b] = 1
            for w in worst_k30[i,:]:
                trades_k30[i][w] = -1
            for b in best_k35[i,:]:
                trades_k35[i][b] = 1
            for w in worst_k35[i,:]:
                trades_k35[i][w] = -1
            for b in best_k40[i,:]:
                trades_k40[i][b] = 1
            for w in worst_k40[i,:]:
                trades_k40[i][w] = -1
            for b in best_k45[i,:]:
                trades_k45[i][b] = 1
            for w in worst_k45[i,:]:
                trades_k45[i][w] = -1
            for b in best_k50[i,:]:
                trades_k50[i][b] = 1
            for w in worst_k50[i,:]:
                trades_k50[i][w] = -1
        if n_stocks > 150:
            for b in best_k75[i,:]:
                trades_k75[i][b] = 1
            for w in worst_k75[i,:]:
                trades_k75[i][w] = -1
        if n_stocks > 200:
            for b in best_k100[i,:]:
                trades_k100[i][b] = 1
            for w in worst_k100[i,:]:
                trades_k100[i][w] = -1

    if n_stocks > 200:
        return [torch.Tensor(trades_kV), torch.Tensor(trades_k1), torch.Tensor(trades_k5), \
               torch.Tensor(trades_k10), torch.Tensor(trades_k15), torch.Tensor(trades_k20), \
               torch.Tensor(trades_k25), torch.Tensor(trades_k30), torch.Tensor(trades_k35), \
               torch.Tensor(trades_k40), torch.Tensor(trades_k45), torch.Tensor(trades_k50), \
               torch.Tensor(trades_k75), torch.Tensor(trades_k100)]
    elif n_stocks > 150:
        return [torch.Tensor(trades_kV), torch.Tensor(trades_k1), torch.Tensor(trades_k5), \
               torch.Tensor(trades_k10), torch.Tensor(trades_k15), torch.Tensor(trades_k20), \
               torch.Tensor(trades_k25), torch.Tensor(trades_k30), torch.Tensor(trades_k35), \
               torch.Tensor(trades_k40), torch.Tensor(trades_k45), torch.Tensor(trades_k50), \
               torch.Tensor(trades_k75)]
    elif n_stocks > 100:
        return [torch.Tensor(trades_kV), torch.Tensor(trades_k1), torch.Tensor(trades_k5), \
               torch.Tensor(trades_k10), torch.Tensor(trades_k15), torch.Tensor(trades_k20), \
               torch.Tensor(trades_k25), torch.Tensor(trades_k30), torch.Tensor(trades_k35), \
               torch.Tensor(trades_k40), torch.Tensor(trades_k45), torch.Tensor(trades_k50)]
    elif n_stocks > 50:
        return [torch.Tensor(trades_kV), torch.Tensor(trades_k1), torch.Tensor(trades_k5), \
               torch.Tensor(trades_k10), torch.Tensor(trades_k15), torch.Tensor(trades_k20), \
               torch.Tensor(trades_k25)]
    elif n_stocks > 30:
        return [torch.Tensor(trades_kV), torch.Tensor(trades_k1), torch.Tensor(trades_k5), \
               torch.Tensor(trades_k10), torch.Tensor(trades_k15)]
    elif n_stocks > 5:
        return [torch.Tensor(trades_kV), torch.Tensor(trades_k1)]


def trade_on_prediction(pred, price_before, price_actual, pred_type):
    if pred_type == 'price':
        prediction = (pred - price_before) / price_before
        true = (price_actual - price_before) / price_before
    if pred_type == 'ret':
        prediction = pred
        true = (price_actual - price_before) / price_before
    if pred_type == 'binclass':
        prediction = pred
        true = (price_actual - price_before) / price_before
    if pred_type == 'binclass_ret':
        prediction = pred
        true = (price_actual - price_before) / price_before

    trades_list =get_trades(prediction, true, pred_type)

    true = torch.Tensor(true.reshape((len(true), len(true[0]))))

    DayReturn_list = []
    long_dayReturn_list = []
    short_dayReturn_list = []
    longshort_dayReturn_list = []
    long_meanDayReturn_list = []
    short_meanDayReturn_list = []
    longshort_meanDayReturn_list = []

    trV_list = [3, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100]
    trV = 0
    for trades in trades_list:
        #calculate return from individual trades
        trades = trades.reshape((len(trades),len(trades[0])))
        long, short = trades.clone(), trades.clone()
        long[long < 0] = 0
        short[short > 0] = 0
        longtradeReturn = true.mul(long)
        shorttradeReturn = true.mul(short)
        #average them per day
        long_dayReturn = torch.sum(longtradeReturn, dim=1) / trV_list[trV]
        short_dayReturn = torch.sum(shorttradeReturn, dim=1) / trV_list[trV]
        longshort_dayReturn = (long_dayReturn + short_dayReturn) / 2
        #average them oveer window (if applicable)
        longshort_dayReturn_list.append(longshort_dayReturn)
        long_dayReturn_list.append(long_dayReturn)
        short_dayReturn_list.append(short_dayReturn)
        longshort_meanDayReturn = torch.mean(longshort_dayReturn, dim=0)
        long_meanDayReturn = torch.mean(long_dayReturn, dim=0)
        short_meanDayReturn = torch.mean(short_dayReturn, dim = 0)
        longshort_meanDayReturn_list.append(longshort_meanDayReturn)
        short_meanDayReturn_list.append(short_meanDayReturn)
        long_meanDayReturn_list.append(long_meanDayReturn)

        trV = trV + 1

    DayReturn_list = torch.cat([torch.Tensor(longshort_meanDayReturn_list), torch.Tensor(long_meanDayReturn_list), torch.Tensor(short_meanDayReturn_list)])
    return np.array(DayReturn_list * 100), np.array(trades_list[1])

