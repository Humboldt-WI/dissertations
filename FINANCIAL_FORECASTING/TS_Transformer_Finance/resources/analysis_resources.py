import numpy as np
from TS_Transformer_Finance.resources import TST_resources as tst

def get_d_return_idx(how):
    if how == 'calculate':
        time_path = 'resources\Timeline_010190_311220.csv'
        data_path = 'resources\SP500_Price_Inputdata.csv'

        close_idx_stocks = tst.load_data(data_path, time_path,0,8088,12)[0].squeeze() #load eikon dataset from csv. start, end, #stocks
        d_return_idx_stocks = np.zeros_like(close_idx_stocks)
        d_return_idx_stocks[1:] = (close_idx_stocks[1:] / close_idx_stocks[:len(close_idx_stocks)-1]) - 1
        d_return_idx = d_return_idx_stocks.mean(axis=1)
        np.savetxt(f'resources\Index_d_return.csv', d_return_idx, delimiter=',')
        return d_return_idx

    if how == 'load':
        d_return_idx = np.loadtxt(f'resources\Index_d_return.csv', delimiter=',')
        return d_return_idx