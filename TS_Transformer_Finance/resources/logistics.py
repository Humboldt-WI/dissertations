import numpy as np
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import csv
from scipy.ndimage.filters import uniform_filter1d



# GET INPUT DATA

def load_data(data_path, time_path, start=0, end=8217, n_stocks=530):
    ''' Loads dataset from specified location and performas initial cleaning for missings.

    :param data_path:   [str] location datafile
    :param time_path:   [str] location timeline file
    :param start:       [int] first day to be included in experiment data (def: 0)
    :param end:         [int] last day to be included in experiment data (max = def: 8217)
    :param n_stocks:    [int] number if stocks to be included in experiment data (def: 530)
    :return:            [array] experiment dataset
    '''

    dataset = []
    with open(data_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            dataset.append(row)
    dataset = np.array(dataset)[start:end+1,:n_stocks].reshape((end-start+1,n_stocks, 1))

    dataset[dataset=='NA'] = np.nan
    dataset = dataset.astype('float32')

    def remove_missing(x):
        ''' Removes zeros from dataset.
            Zeros ('0') in the price data are imputed with next price unequal zero for each stock.
            NAs are not considered.

        :param x:   [array] raw input dataset
        :return:    [array] cleaned input dataset
        '''
        for stock in range(len(x[0])):
            if x[0,stock] == 0:
                for day in range(len(x)):
                    if x[day,stock]!=0:
                        x[0,stock] = x[day,stock]
                        break
                    else:
                        x[0,stock] = 0.01
            for day in range(len(x)):
                if x[day,stock] == 0:
                    x[day,stock] = x[day - 1,stock]
        return x
    dataset = remove_missing(dataset)

    timeline = []
    with open(time_path) as csvTimeFile:
        csvReader = csv.reader(csvTimeFile)
        for row in csvReader:
            timeline.append(row)
    timeline = np.array(timeline)[:, start:end+2].reshape((2,end-start+2))[1, 1:]

    return dataset, timeline

''' Normalization functions for input data.'''
#BEFORE: norm(x)
def normalize(x):
    ''' Min-Mayx Normalization.

    :param x:   [array] data
    :return:    [array] normalized data
    '''
    #minimum = np.nanmin(x) #np.nanmin(x, axis=(0,1))
    #maximum = np.nanmax(x) #np.nanmax(x, axis=(0,1))
    #return (x - minimum) / (maximum - minimum + np.finfo(np.float32).eps), minimum, maximum
    minimum = np.nanmin(x, axis=(1, 2))  # np.nanmin(x) #np.nanmin(x, axis=(0,1))
    maximum = np.nanmax(x, axis=(1, 2))  # np.nanmax(x) #np.nanmax(x, axis=(0,1))
    minimum = np.repeat(np.repeat(minimum.reshape((len(minimum),1,1)), (len(x[0])), axis=1),len(x[0,0]),axis=2)
    maximum = np.repeat(np.repeat(maximum.reshape((len(maximum),1,1)), (len(x[0])), axis=1),len(x[0,0]),axis=2)
    return (x - minimum) / (maximum - minimum + np.finfo(np.float32).eps), minimum[:,0,0], maximum[:,0,0]


def normalize_with(x, minimum, maximum):
    ''' MIN-Max Normalization with provided scale'''
    minimum = np.repeat(np.repeat(minimum.reshape((len(minimum), 1, 1)), (len(x[0])), axis=1), len(x[0, 0]), axis=2)
    maximum = np.repeat(np.repeat(maximum.reshape((len(maximum), 1, 1)), (len(x[0])), axis=1), len(x[0, 0]), axis=2)
    return (x - minimum) / (maximum - minimum), minimum, maximum


def ret_normalize(x):
    ''' Fixed-scale Min-Max Normalization (for return data).

    :param x:   [array] return-data
    :return:    [array] normalized data
    '''
    minimum = -0.15
    maximum = 0.15
    return (x - minimum) / (maximum - minimum), minimum, maximum


def reverse_normalize(x, minimum, maximum):
    ''' Reverse normalization operations.

    :param x:       [array] normalized data
    :param minimum: minimum from initial normalization
    :param maximum: maximum from initial normalization
    :return:        [array] rescaled data
    '''
    return (x * (maximum - minimum) + minimum)


# CALCULATE INPUT FEATURES
''' Functions to convert daily closing prices into various other feature types.
    Types included:
        - day-to-day return
        - daily ranking
        - moving return average
        - momentum
        - binary classification'''

def price_to_ret(x):
    ''' day-to-day return

    :param x:   [array] daily closing prices
    :return:    [array] daily returns
                (ret_t = (price_t - price_t-1) / price_t-1)
    '''

    d_time = len(x)
    yesterday = np.zeros_like(x)
    yesterday[1:] = x[:d_time - 1]
    yesterday[0] = x[0]
    ret = (x - yesterday) / yesterday
    return ret

def price_to_ranking(x):
    ''' daily ranking

    :param x:   [array] daily closing prices
    :return:    [array] daily ranking from best to worst performing stock
                (based on day's return)
    '''

    d_time, d_stock, _ = x.shape
    ret = price_to_ret(x)
    ranking = np.empty_like(ret)
    for i in range(0,d_time):
        temp = np.argsort(ret[i],axis=0).reshape(d_stock)
        if ret[i].min() == ret[i].max():
            temp = np.zeros_like(temp) + int(d_stock/2)
        ranking[i][temp] = np.arange(d_stock).reshape((d_stock, 1))
    ranking = ranking / d_stock
    return ranking

def price_to_average(x, days):
    '''moving average of daily returns

    :param x:       [array] daily closing prices
    :param days:    [int] number of past days to average
    :return:        [array] moving average of return over past # days
    '''
    d_time, d_stock, _ = x.shape
    X = np.nan_to_num(x, copy=True)
    average = uniform_filter1d(X.astype(np.float32), size=days, axis=0, mode='nearest', origin=int(days/2))
    min = np.nanmin(average)
    max = np.nanmax(average)
    average = (average - min) / (max - min)
    return average

def price_to_momentum(x, days):
    ''' momentum

    :param x:       [array] daily closing prices
    :param days:    [int] number of past days for current side
    :return:        [array] normalized momentum scores
                    (momentum = (current_avg - reference_avg))
    '''

    d_time, d_stock, _ = x.shape
    ret = price_to_ret(x)
    ret = np.nan_to_num(ret, copy=True)
    lastdays = uniform_filter1d(ret.astype(np.float32), size=days, axis=0, mode='nearest', origin=int(days/2))
    reference = uniform_filter1d(ret.astype(np.float32), size=3*days, axis=0, mode='nearest', origin=int((3*days)/2))
    momentum = lastdays - reference
    min = np.nanmin(momentum)
    max = np.nanmax(momentum)
    momentum = (momentum - min) / (max - min)
    return momentum

def price_to_binclass(x):
    ''' binary classification

    :param x:   [array] dail closing prices
    :return:    [array] binary classification
                ret_i >= median_ret: 1, 0 otherwise
    '''

    d_time, d_stock, _ = x.shape
    ret = price_to_ret(x)
    ret = np.nan_to_num(ret, copy=True)
    median = np.nanmedian(ret, axis=1)
    binclass = np.zeros_like(x)
    for time in range(d_time):
        binclass[time][ret[time]>median[time]] = 1
        binclass[time][ret[time] < median[time]] = 0
    return binclass

#MANIPULATE INPUT DATASET
''' Functions to slice input dataset in sequences for model feeds.'''

def slice_dataset(dataset, window_size=80, mode='None;0'):
    ''' Slices and copies dataset into sequences for feedint to the predictive model

    :param dataset:     [array] input dataset
    :param look_back:   [int] size of look back window (def: 80)
    :param mode:        [str] look back type (None: t-79 - t0;
                        sparse;int: add elements before t-79 + t-79 - t0)(def: None;0)
    :return:            [array] seuquenced input data
    '''
    mode, add_points = str.split(mode, ';')
    a,b,c = dataset.shape
    dataX, dataY = [], []
    for i in range(len(dataset) - window_size):
        a = dataset[i:(i + window_size), :, :]
        dataX.append(a)
        dataY.append(dataset[i + window_size, :, :].reshape(1,b,c))

    if mode == 'sparse':
        points = []
        for i in range(int(add_points)):
            points.append(int(math.exp(i+1)/(i*1.5+2)))
        points = np.array(points)

        for point in points:
            data_add = []
            for i in range(len(dataset) - window_size):
                if i > point:
                    data_add.append(dataset[i-point, :, :])
                else:
                    data_add.append(dataset[0, :, :])
            data_add = np.array(data_add).reshape((len(dataX), 1, b, c))
            dataX = np.concatenate((data_add, dataX), axis=1)

    return np.array(dataX), np.array(dataY)


def filter_unlisted(Xdata, Ydata, PXdata, PYdata):
    ''' Selects those sequences from the data, that don't contain NAs.
        Ensures stocks, that are not yet listed and traded are not included in the model input sequences for training/prediction.

    :param Xdata: [array] model input sequences
    :param Ydata: [array] model input sequences
    :return:      [arrays] selected model input sequences
                  [array]  included stocks (indices)
    '''

    fully_listed_stocks = ~np.isnan(Xdata[0, 0, :, 0]) & ~np.isnan(Ydata[0, 0, :, 0]) & ~np.isnan(PXdata[-1,-1,:,0]) & ~np.isnan(PYdata[-1,-1,:,0])
    return Xdata[:, :, fully_listed_stocks], Ydata[:, :, fully_listed_stocks], PXdata[:,:,fully_listed_stocks], PYdata[:,:,fully_listed_stocks],fully_listed_stocks


def prepare_data(in_data, var_order):

    d_time, d_stock, _ = in_data.shape
    out_data = np.empty((d_time,d_stock,len(var_order)))

    for var in range(len(var_order)):
        if var_order[var] == 'price':
            out_data[:,:,var] = in_data.reshape((d_time, d_stock))
        if var_order[var] == 'ret':
            out_data[:,:,var] = price_to_ret(in_data[:,:].reshape((d_time, d_stock,1))).reshape((d_time, d_stock))
        if var_order[var] == 'ranking':
            out_data[:, :, var] = price_to_ranking(in_data[:, :].reshape((d_time, d_stock,1))).reshape((d_time, d_stock))
        if var_order[var] == 'average':
            out_data[:, :, var] = price_to_average(in_data[:, :].reshape((d_time, d_stock, 1)),15).reshape((d_time, d_stock))
        if var_order[var] == 'momentum':
            out_data[:, :, var] = price_to_momentum(in_data[:, :].reshape((d_time, d_stock,1)),5).reshape((d_time, d_stock))
        if var_order[var] == 'binclass':
            out_data[:, :, var] = price_to_binclass(in_data[:, :].reshape((d_time, d_stock,1))).reshape((d_time, d_stock))

    return out_data


def prepare_step(dataset, var_order, day, window_size, block_size, mode='no;5', trg_in_src=False):

    # SELECT BLOCK PERIOD
    train_size = block_size
    test_size = min(block_size, dataset.shape[0] - day)
    start_train, end_test = day - train_size, day + test_size
    train, test = dataset[start_train:day, :], dataset[day - window_size:end_test - window_size, :]
    X_day = start_train
    Y_day = X_day + 1

    # SLICE DATA
    trainX, trainY = slice_dataset(train, window_size=window_size, mode=mode)  # [time, window, stock]
    testX, testY = slice_dataset(test, window_size=window_size, mode=mode)

    # FILTER NAs (UNLISTED STOCKS)
    trainX, trainY, testX, testY, train_stock_idx = filter_unlisted(trainX, trainY, testX, testY)
    test_stock_idx = train_stock_idx

    # RESHAPE
    # TRAIN SET
    X0 = trainX[0:-2]
    Y0 = trainX[1:-1]
    #X is input to encoder, Y is activation to decoder
    d_time, d_wind, d_stock, d_var = X0.shape
    X0 = X0.transpose(2, 0, 1, 3).reshape(d_stock, d_time, d_wind, d_var).astype(
        np.float32)
    Y0 = Y0.transpose(2, 0, 1, 3).reshape(d_stock, d_time, d_wind, d_var).astype(
        np.float32)

    #PREDICTION SET
    P_X0 = testX[0:-2, :, :, :]
    P_Y0 = testX[1:-1, :, :, :]
    P_X_day = day - 1
    P_Y_day = P_X_day + 1

    P_d_time, P_d_wind, P_d_stock, P_d_var = P_X0.shape
    P_X0 = P_X0.transpose(2, 0, 1, 3).reshape(P_d_stock, P_d_time, P_d_wind, P_d_var).astype(
        np.float32)
    P_Y0 = P_Y0.transpose(2, 0, 1, 3).reshape(P_d_stock, P_d_time, P_d_wind, P_d_var).astype(
        np.float32)  #

    # NORMALIZATION
    var_types = ['price', 'price2', 'ret', 'ranking', 'average', 'momentum', 'binclass']
    minimum = np.empty((d_var,d_stock))
    maximum = np.empty((d_var,d_stock))
    for var in range(len(var_order)):
        if var_order[var] in ['price','price2','ret']:
            X0[:,:,:,var], minimum[var], maximum[var] = normalize(X0[:,:,:,var])
            Y0[:,:,:,var],_,_ = normalize_with(Y0[:,:,:,var],minimum[var],maximum[var])
            P_X0[:,:,:,var],_,_ = normalize_with(P_X0[:,:,:,var],minimum[var],maximum[var])
            P_Y0[:,:,:,var],_,_ = normalize_with(P_Y0[:,:,:,var], minimum[var], maximum[var])

    if var_order[-1] == 'binclass':
        X0=np.concatenate((X0[:,:,:,:], (np.ones_like(X0[:,:,:,-1]) - X0[:,:,:,-1]).reshape((d_stock,d_time,d_wind,1))), axis=-1).squeeze()
        Y0=np.concatenate((Y0[:, :, :,:], (np.ones_like(Y0[:, :, :, -1]) - Y0[:, :, :, -1]).reshape((d_stock,d_time,d_wind,1))), axis=-1).squeeze()
        P_X0=np.concatenate((P_X0[:, :, :,:], (np.ones_like(P_X0[:, :, :, -1]) - P_X0[:, :, :, -1]).reshape((d_stock,d_time,d_wind,1))), axis=-1).squeeze()
        P_Y0=np.concatenate((P_Y0[:, :, :,:], (np.ones_like(P_Y0[:, :, :, -1]) - P_Y0[:, :, :, -1]).reshape((d_stock,d_time,d_wind,1))), axis=-1).squeeze()

    return X0, Y0, P_X0, P_Y0, X_day, Y_day, P_X_day, P_Y_day, minimum[-1], maximum[-1], train_stock_idx, test_stock_idx