import numpy as np
import torch
import resources.evaluation_resources as ev
from resources.trade_resources import trade_on_prediction
import resources.logistics as log

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(7)

"""________PARAMETERS________"""
EID = 'Test'                    # Identifier for experiment run. Used to identify results for analysis.

run = 'single'                  # Experiment to run. The same experiment as transformer/LSTM but with random prediction, or
                                # an aggregated experiment just reuturning mean daily portfolio returns.
                                # Available:    - 'single' detailed single data run output (return, accuracy, trades)
                                #               - 'multiple' aggregated portfolio returns for specified number of data runs.
""" model parameters """
window_size = 1                 # (Legacy) Window size for lookback in sequences. Keep at 1
data_runs = 10000               # Only necessary for run='multiple'. Number of data runs (indiv. mini-experiments).
""" sequence parameters """
block_size = 10                 # Total number of days in training (prediction) set.
                                # Number of sequences per stock ~ block_size - window_size.
step_size = 7                   # Days to predict per step. Must be smaller than block_size - window_size - 2.
""" data parameters """
start = 0                       # For partial execution. First day to predict. Automatic adaption if smaller than block_size.
end = 8217                      # For partial execution. Last day to predict.
var_order = ['price','binclass']# Data type in input and target sequences.
                                # For run='single' choose ['ret'], for 'multiple' choose ['price','binclass']
                                # Format: ['input 1', 'input 2', ..., 'target']
                                # Available:    - 'ret' daily returns
                                #               - 'price' daily closing prices
                                #               - 'average' mean of last 15 days daily returns (norm.)
                                #               - 'momentum' mean of last 5 days minus last 15 days daily returns (norm.)
                                #               - 'ranking' daily ordinal ranking of stocks
                                #               - 'binclass' daily binary classification of stocks in above/below median daily return
""" location parameters """
data_path = 'resources\SP500_Price_Inputdata.csv'   # Path to raw input data.
time_path = 'resources\Timeline_010190_300621.csv'  # Path to input timeline.
#For loading and saving inout and result data it is important, that the directory structure from this repository is mirrored in the execution environment.


""" ------ RUN EXPERIMENT ------ """


""" --- GET DATA AND STORAGE --- """

dataset, timeline = log.load_data(data_path, time_path,0,8217,530)
price_data = dataset[:,:,0]
dataset = log.prepare_data(dataset, var_order)

predictions = np.zeros_like(dataset)[:,:,0]
traded_stocks = np.zeros_like(dataset)[:,:,0]
return_history = np.zeros((len(dataset),42))
accuracy = np.zeros((len(dataset),5))
pred_iterations_counter = 0

m_init_counter = 0
model = []
model_opt = []

""" --- SINGLE DETAILED EXPERIMENT --- """

if run == 'single':
    for day in range(start, dataset.shape[0], step_size):
        if day >= block_size:

            """ --- PREPROCESSING --- """

            X0, Y0, P_X0, P_Y0, X_day, Y_day, P_X_day, P_Y_day, minimum, maximum, train_stock_idx, test_stock_idx = log.prepare_step(
                dataset, var_order, day, window_size, block_size)
            d_stock, d_time, d_var = X0.shape

            """ --- PREDICTION --- """

            output = np.zeros_like(P_Y0)[:,:,-2]
            dicerollprediction = np.random.rand(len(output), len(output[0]))
            output = dicerollprediction

            for place in range(step_size):
                x_day = P_X_day + place
                y_day = x_day + 1

                # RESCALE PREDICTION
                pred = output[:,place]
                actual_target = P_Y0[:,place,-2]

                """ ----- TRADE ----- """

                price_before = price_data[x_day, test_stock_idx]
                price_actual = price_data[y_day, test_stock_idx]

                # TRADE MODEL
                trade_results, trades = trade_on_prediction(np.array(pred.reshape((1, len(pred), 1))),
                                                            price_before.reshape(
                                                                (1, price_before.size, price_before[0].size)),
                                                            price_actual.reshape(
                                                                (1, price_actual.size, price_actual[0].size)),
                                                            var_order[-1])
                pred_day = day - 1 + place  # pred_day = day + look_back - 1 + place
                predictions[pred_day, test_stock_idx] = pred.reshape(len(pred))
                traded_stocks[pred_day, test_stock_idx] = trades
                return_history[pred_day, :len(trade_results)] = trade_results

                pred_iterations_counter += 1
                avg_return = return_history.sum(axis=0) / (pred_iterations_counter)

                # ACCURACY
                accuracy[pred_day] = ev.pred_acc(np.array(pred.reshape((1, len(pred), 1))),
                                                 price_before.reshape((1, price_before.size, price_before[0].size)),
                                                 price_actual.reshape((1, price_actual.size, price_actual[0].size)),
                                                 var_order[-1])
                avg_accuracy = accuracy.sum(axis=0)[0] / (pred_iterations_counter)

                print("Day:        ", pred_day,
                      ",  first five stocks, return order: c,c_,l,l_,s,s_, acc. order: acc,pTtT,pFtF,pTtF,pFtT",
                      "\nactual:     ", actual_target[:5], "\nprediction: ", pred.reshape(len(pred))[:5],
                      "\naccuracy:   ", accuracy[pred_day], " avg. acc.: ", avg_accuracy, "\nreturn:     ",
                      trade_results[:5], "\navg. return:", avg_return[avg_return != 0][:5])

                """ ----- SAVE ----- """

                np.savetxt(f'experiment/return_' + str(EID) + '.csv', return_history, delimiter=',')
                np.savetxt(f'experiment/accuracy_' + str(EID) + '.csv', accuracy, delimiter=',')
                np.savetxt(f'experiment/trades_' + str(EID) + '.csv', traded_stocks, delimiter=',')


""" --- MULTIPLE AGGREGATED EXPERIMENT --- """

if run == 'multiple':

    """ --- PREDICTION --- """

    returns = dataset.reshape((len(dataset),len(dataset[0])))
    portfolio_return = np.zeros(data_runs)
    for j in range(0,data_runs):
        k1_trades = np.zeros_like(returns)
        for i in range(240,len(k1_trades)):
            trades = np.random.randint(low=0, high=530, size=2)
            k1_trades[i,trades[0]] = -1
            k1_trades[i,trades[1]] = 1

        """ ----- TRADE ----- """

        trade_returns = np.multiply(returns,k1_trades)
        return_history = np.nansum(trade_returns, axis=1)/2
        mean_return = np.nanmean(return_history[240:8088]) * 100
        portfolio_return[j] = mean_return
        print(str(j) + ': ' + str(mean_return))

    """ ----- SAVE ----- """

    np.savetxt(f'experiment/Monkey_Portfolio_returns' + str(EID) + '.csv', portfolio_return, delimiter=',')
    print(np.nanmean(portfolio_return))

print('Experiment run finished.')