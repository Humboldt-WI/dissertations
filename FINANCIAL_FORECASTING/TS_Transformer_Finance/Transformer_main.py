import numpy as np
import torch
import torch.nn as nn
import copy
import resources.evaluation_resources as ev
from resources.trade_resources import trade_on_prediction
import resources.logistics as log
import resources.TST_resources as tst
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(1391415)

"""________PARAMETERS________"""

EID = 'Test'                    # Identifier for experiment run. Used to identify results for analysis.
""" model parameters """
window_size = 20                # Window size for lookback in sequences. (def. 80; int)
V = 200                         # Value, Query, Key dimension ~ vocab size. (def: 200; int)
N = 6                           # Encoder and decoder layers in stack. (def: 4; int)
dropout = 0.1                   # Dropout probability. (def: 0.1, float)
convolve = 2                    # Convolution kernel in MHA-module. (def: 3; int). 0 ~ no convolution layer
pe_type = 'global'              # Positional encoding. (str)
                                # Available:    - relative (def)
                                #               - None
                                #               - global
                                #               - weekday
                                #               - yearday
attention_type = 'full'         # Full or sparse attention conncetions e.g. [5,3,1,0]. Which elements of the full q,v,k tensors should be used. No additionals.
                                # Available:    - 'full': full attention
                                #               - [5,3,1,0]: with the list elements beeing the indexes of sequence elements from its end, to include. Order as depiced.
mode = 'no;5'                   # Sparse, ... ; no of add. datapoints. (def. 'no;5' ~ no additional; 'sparse; int')
""" training parameters """
criterion = nn.MSELoss()        # Torch loss criterion. (def. nn.MSELoss())
learning = 0.001                # Learning rate. (def. 0.001; float)
epochs = 20                     # Epochs per step. (def. 20; int)
patience = 2                    # Patience for early stopping. (def. 2; int)
nbatch = 1                      # Batch numbers. Number of batches to divide stocks into. (int)
minibatch = True                # Overwrites nbatch and sets it to nstock resulting in minibatches of size 1. (def. True, boolean)
m_init_freq = 1                 # Model compilation frequency. After how many steps model is recompiled. (def. 1; int)
                                # (0: model is only compiled once; 1,2,...: model is recompiled every n steps.)
""" sequence parameters """
block_size = 200                # Total number of days in training (prediction) set. (def. 320; int)
                                # Number of sequences per stock ~ block_size - window_size.
step_size = 40                  # Days to predict per step. Must be smaller than block_size - window_size-2. (def. 40; int)
""" data parameters """
start = 0                       # For partial execution. First day to predict. Automatic adaption if smaller than block_size. (def. 0; int)
end = 2000                      # For partial execution. Last day to predict. (def. 8217; int)
var_order = ['price','price']   # Data type in input and target sequences.
                                # Format: ['input 1', 'input 2', ..., 'target']
                                # Available:    - 'ret' daily returns
                                #               - 'price' daily closing prices
                                #               - 'average' mean of last 15 days daily returns (norm.)
                                #               - 'momentum' mean of last 5 days minus last 15 days daily returns (norm.)
                                #               - 'ranking' daily ordinal ranking of stocks
                                #               - 'binclass' daily binary classification of stocks in above/below median daily return
trg_in_src = False              # Legacy parameter. Keep False.
d_tgt = 1                       # Legacy parameter. Keep 1. Adjust if implementing additional multivariate encoded target type.
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

for day in range(start, dataset.shape[0], step_size):
    if day >= block_size:

        """ --- PREPROCESSING --- """

        X0, Y0, P_X0, P_Y0, X_day, Y_day, P_X_day, P_Y_day, minimum, maximum, train_stock_idx, test_stock_idx = log.prepare_step(dataset, var_order, day, window_size, block_size, mode)
        d_stock, d_time, d_wind, d_var = X0.shape

        """ --- BUILD MODEL --- """

        if trg_in_src == False:
            d_var = d_var - 1
        if var_order[-1] == 'binclass':
            d_var = d_var - 1

        if tst.m_init_switch(m_init_freq, m_init_counter):
            model = tst.make_model(V, V, d_wind, d_var, d_tgt, timeline=timeline, N=N, dropout=dropout,
                                  convolve=convolve, d_time=X0.shape[1], pe_type=pe_type, attention_type=attention_type).to(device)
            model_opt = tst.NoamOpt(model.src_embed[0].d_model, 10, 400,torch.optim.Adam(model.parameters(), lr=learning, betas=(0.9, 0.98), eps=1e-9))
        m_init_counter = m_init_counter + 1

        """     TRAIN     """
        if minibatch == True:
            nbatch = d_stock

        # EARLY STOPPING
        best_val_loss = 999999
        patience_counter = 0

        for epoch in range(epochs):  # (300)
            # model.train()
            loss, val_loss = tst.run_epoch(tst.data_gen(V, 999, nbatch, X0, Y0, X_day, Y_day, var_order), model.to(device),
                      tst.SimpleLossCompute(model.generator, model.out_generator, criterion, model_opt),learning, nbatch, d_stock)
            #X_day is index of first day in X0
            print('Epoch: '+str(epoch+1)+' Loss: '+str(loss)+' Validation Loss: '+str(val_loss))
            if val_loss < best_val_loss:
                best_val_loss =  val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            elif val_loss >= best_val_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        model.load_state_dict(best_model_state)

        """ --- PREDICTION --- """

        X0 = P_X0
        Y0 = P_Y0
        X_day = P_X_day
        Y_day = P_Y_day
        d_stock = X0.shape[0]
        x0 = X0
        y0 = Y0

        output = np.zeros_like(x0)[:,:,-1,0]
        for i, batch in enumerate(tst.data_gen(V, 999, nbatch, x0, y0, X_day, Y_day, var_order)):
            out = model.forward(batch.src.to(device), batch.trg.to(device),
                                batch.src_mask.to(device), batch.trg_mask.to(device), batch.X_day, batch.Y_day).to(device)
            a, b, c, d = batch.trg.shape
            out = model.generator(out.reshape((a, b, c, -1, d)).permute(0,1,2,4,3))
            out = model.out_generator(out.permute(0,1,2,4,3)).permute(0,1,2,4,3)
            out = torch.sum(out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3], -1), (-1))
            output[i] = out[:,:,-1,0].detach()

        for place in range(step_size):
            x_day = X_day + place
            y_day = x_day + 1

            # RESCALE PREDICTION
            pred = output[:,place]
            actual_target = y0[:,place,-1,-1]

            if var_order[-1] in ['price','ret']:
                pred = log.reverse_normalize(pred, minimum, maximum)
                actual_target = log.reverse_normalize(actual_target, minimum, maximum)

            """ ----- TRADE ----- """

            price_before = price_data[x_day, test_stock_idx]
            price_actual = price_data[y_day, test_stock_idx]

            #TRADE MODEL
            trade_results, trades = trade_on_prediction(np.array(pred.reshape((1, len(pred),1))),
                                                price_before.reshape((1, price_before.size, price_before[0].size)),
                                                price_actual.reshape((1, price_actual.size, price_actual[0].size)),
                                                var_order[-1])
            pred_day = day - 1 + place #pred_day = day + look_back - 1 + place
            predictions[pred_day,test_stock_idx] = pred.reshape(len(pred))
            traded_stocks[pred_day, test_stock_idx] = trades
            return_history[pred_day, :len(trade_results)] = trade_results

            pred_iterations_counter += 1
            avg_return = return_history.sum(axis=0) / (pred_iterations_counter)

            #ACCURACY
            accuracy[pred_day] = ev.pred_acc(np.array(pred.reshape((1, len(pred),1))),
                                             price_before.reshape((1, price_before.size, price_before[0].size)),
                                             price_actual.reshape((1, price_actual.size, price_actual[0].size)),
                                                var_order[-1])
            avg_accuracy = accuracy.sum(axis=0)[0] / (pred_iterations_counter)

            print("Day:        ", pred_day, ",  first five stocks, return order: c,c_,l,l_,s,s_, acc. order: acc,pTtT,pFtF,pTtF,pFtT", "\nactual:     ", actual_target[:5], "\nprediction: ", pred.reshape(len(pred))[:5], "\naccuracy:   ", accuracy[pred_day], " avg. acc.: ", avg_accuracy, "\nreturn:     ", trade_results[:5], "\navg. return:", avg_return[avg_return!=0][:5])

            """ ----- SAVE ----- """

            np.savetxt(f'experiment/return_' + str(EID) + '.csv', return_history, delimiter=',')
            np.savetxt(f'experiment/accuracy_' + str(EID) + '.csv', accuracy, delimiter=',')
            np.savetxt(f'experiment/trades_' + str(EID) + '.csv', traded_stocks, delimiter=',')

print('Experiment run finished.')
