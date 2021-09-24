import numpy as np
import torch
from tensorflow import keras as ks
import resources.logistics as log
import resources.evaluation_resources as ev
from resources.trade_resources import trade_on_prediction
from resources.TST_resources import m_init_switch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(1391415)

"""     LSTM - Model     """

""" --- DEFINE PARAMETERS --- """

EID = 'Test'                                # Identifier for experiment run. Used to identify results for analysis.
""" model parameters """
window_size = 80                            # Window size for lookback in sequences. (def. 80; int)
dropout = 0.1                               # Dropout probability. (def: 0.1, float)
""" training parameters """
criterion = ks.losses.BinaryCrossentropy()  # Keras loss criterion. (def. BinaryCrossentropy())
learning = 0.001                            # Learning rate. (def. 0.001; float)
epochs = 20                                 # Epochs per step. (def. 20; int)
patience = 3                                # atience for early stopping. (def. 2; int)
m_init_freq = 1                             # Model compilation frequency. After how many steps model is recompiled. (def. 1; int)
                                            # (0: model is only compiled once; 1,2,...: model is recompiled every n steps.)
""" sequence parameters """
block_size=320                              # Total number of days in training (prediction) set. (def. 320; int)
                                            # Number of sequences per stock ~ block_size - window_size.
step_size = 40                              # Days to predict per step. Must be smaller than block_size - window_size-2. (def. 40; int)
""" data parameters """
start = 0                                   # For partial execution. First day to predict. Automatic adaption if smaller than block_size. (def. 0; int)
end = 8217                                  # For partial execution. Last day to predict. (def. 8217; int)
var_order = ['ret','binclass']              # Data type in input and target sequences.
                                            # Format: ['input 1', 'input 2', ..., 'target']
                                            # Available:    - 'ret' daily returns
                                            #               - 'price' daily closing prices
                                            #               - 'average' mean of last 15 days daily returns (norm.)
                                            #               - 'momentum' mean of last 5 days minus last 15 days daily returns (norm.)
                                            #               - 'ranking' daily ordinal ranking of stocks
                                            #               - 'binclass' daily binary classification of stocks in above/below median daily return
trg_in_src = False                          # Legacy parameter. Keep False.
""" location parameters """
data_path = 'resources\SP500_Price_Inputdata.csv'   # Path to raw input data.
time_path = 'resources\Timeline_010190_300621.csv'  # Path to input timeline.
#For loading and saving inout and result data it is important, that the directory structure from this repository is mirrored in the execution environment.


""" ------ RUN EXPERIMENT ------ """


""" --- GET DATA AND STORAGE --- """

dataset, timeline = log.load_data(data_path, time_path,0,8217 ,530)
price_data = dataset[:,:,0]

dataset = log.prepare_data(dataset, var_order)

d_time, d_stock, d_var = dataset.shape

predictions = np.zeros_like(dataset)[:,:,0]
return_history = np.zeros((len(dataset),42))
traded_stocks = np.zeros_like(dataset)[:,:,0]
accuracy = np.zeros((len(dataset),5))
m_init_counter = 0
pred_iterations_counter = 0

""" --- BUILD MODEL --- """

# MODEL ARCHITECTURE
if not trg_in_src:
    inputs = ks.Input(shape=(window_size, d_var-1))
else:
    inputs = ks.Input(shape=(window_size, d_var))
lstmLayers = ks.layers.LSTM(25, dropout=dropout, recurrent_activation='sigmoid')(inputs)
outputs = ks.layers.Dense(2, activation='softmax')(lstmLayers)
model = ks.Model(inputs=inputs, outputs=outputs, name='LSTM_benchmark_model')
es_cb = ks.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience)
cb_list = [es_cb]


for day in range(start, dataset.shape[0], step_size):
    if day >= block_size:

        """ --- PREPROCESSING --- """

        X0, Y0, P_X0, P_Y0, X_day, Y_day, P_X_day, P_Y_day, minimum, maximum, train_stock_idx, test_stock_idx = log.prepare_step(
            dataset, var_order, day, window_size, block_size)
        d_stock, d_time, d_wind, d_var = X0.shape
        # X and Y are in order (d_stock,d_time,d_wind,d_var). LSTM needs (d_channels,d_wind,1)

        #stacking all sequences per time-step and stock.
        #(order: d0s0,d0s1,d0s2...d1s0,d1s1,d1s2... in first dimension)
        if var_order[-1]=='binclass':
            X0 = (X0[:,:,:,:-2].transpose((1,0,2,3))).reshape((d_time*d_stock,d_wind,d_var-2),order='A')
            Y0 = (Y0[:,:,:,-2:].transpose((1,0,2,3))).reshape((d_time*d_stock,d_wind,2),order='A')[:,-1,:]
            P_X0 = P_X0[:, :, :, :-2]
            P_Y0 = P_Y0[:, :, -1, -2:]
        elif var_order[-1] in ['ret', 'price']:
            X0 = (X0[:,:,:,:-1].transpose((1,0,2,3))).reshape((d_time*d_stock,d_wind,d_var-1),order='A')
            Y0 = (Y0[:,:,:,-1].transpose((1,0,2))).reshape((d_time*d_stock,d_wind,1),order='A')[:,-1,:]
            P_X0 = P_X0[:, :, :, :-1]
            P_Y0 = P_Y0[:, :, -1, -1:]

        """ --- COMPILE MODEL --- """

        if m_init_switch(m_init_freq, m_init_counter):
            model.compile(loss=criterion,
                          optimizer=ks.optimizers.Adam(learning_rate=learning, clipnorm=True),
                          # ks.optimizers.RMSprop(), ks.optimizers.SGD(), ks.optimizers.Adam() - FischerKrauss use RMSprop, but this is unstable for me often resulting in nan loss. Adams seems to be better here.
                          metrics=[ks.metrics.BinaryCrossentropy(),
                                   'accuracy'])
        m_init_counter = m_init_counter + 1

        """ ------ TRAIN ------ """
        model.fit(X0, Y0, batch_size=d_stock, epochs=epochs, validation_split=0.2, verbose=1,
                                     shuffle=False, callbacks=cb_list)

        """ --- PREDICTION --- """
        d_stock,d_time,d_wind,d_var = P_X0.shape

        for place in range(step_size):
            p_X0 = P_X0[:,place,:,:].reshape(d_stock, d_wind, d_var)
            p_Y0 = P_Y0[:,place,:].reshape(d_stock, 1, -1)

            pred = model.predict(p_X0, batch_size=d_stock)
            pred = np.reshape(pred[:,0], (1, d_stock, 1))

            actual_target = p_Y0[:, :, 0].reshape((1, d_stock, 1))

            # RESCALE PREDICTION
            if var_order[-1] in ['price','ret']:
                pred = log.reverse_normalize(pred.reshape(d_stock), minimum, maximum).reshape((1,d_stock,1))
                actual_target = log.reverse_normalize(actual_target.reshape(d_stock), minimum, maximum).reshape((1,d_stock,1))

            """ ----- TRADE ----- """

            price_actual = price_data[P_Y_day, test_stock_idx]
            price_before = price_data[P_X_day, test_stock_idx]

            #TRADE MODEL
            trade_results, trades = trade_on_prediction(np.array(pred),
                                                price_before.reshape((1, price_before.size, price_before[0].size)),
                                                price_actual.reshape((1, price_actual.size, price_actual[0].size)),
                                                var_order[-1])

            pred_day = P_Y_day
            predictions[P_Y_day,test_stock_idx] = pred.reshape(len(pred[0]))
            traded_stocks[P_Y_day, test_stock_idx] = trades
            return_history[P_Y_day, :len(trade_results)] = trade_results

            pred_iterations_counter += 1
            avg_return = return_history.sum(axis=0) / (pred_iterations_counter)

            #ACCURACY
            accuracy[pred_day] = ev.pred_acc(np.array(pred),
                                             price_before.reshape((1, price_before.size, price_before[0].size)),
                                             price_actual.reshape((1, price_actual.size, price_actual[0].size)),
                                                var_order[-1])
            avg_accuracy = accuracy.sum(axis=0)[0] / (pred_iterations_counter)

            print("Day:        ", pred_day, ",  first five stocks, return order: c,c_,l,l_,s,s_, acc. order: acc,pTtT,pFtF,pTtF,pFtT", "\nactual:     ", actual_target.reshape((len(actual_target[0])))[:5], "\nprediction: ", pred.reshape((len(pred[0])))[:5], "\naccuracy:   ", accuracy[pred_day], " avg. acc.: ", avg_accuracy, "\nreturn:     ", trade_results[:5], "\navg. return:", avg_return[avg_return!=0][:5])

            """ ----- SAVE ----- """

            np.savetxt(f'experiment/return_' + str(EID) + '.csv', return_history, delimiter=',')
            np.savetxt(f'experiment/accuracy_' + str(EID) + '.csv', accuracy, delimiter=',')
            np.savetxt(f'experiment/trades_' + str(EID) + '.csv', traded_stocks, delimiter=',')

print('Experiment run finished.')