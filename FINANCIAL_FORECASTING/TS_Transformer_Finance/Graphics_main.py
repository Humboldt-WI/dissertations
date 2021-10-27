import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set_theme()
import matplotlib.pyplot as plt
from matplotlib import dates
plt.interactive(False)
import pandas as pd




EIDs = [ '1', '2', '3', '4', '5', '6', '7', '8']  #[1, 2, ...]
Names = ['TST     1', 'TST     2', 'TST     3', 'TST     4', 'TST     5', 'TST     6', 'TST     7', 'TST     8']
#Names = ['LSTM  9', 'LSTM 10']
#EIDs = [ '49', '1012.1' ]

#only for ret_acc_1_100
Legend = ['1. TST global PE', '2. TST relative PE', '3. TST weekly-periodic PE', '4. TST no PE',
        '5. TST no convolution', '6. TST sparse attention', '7. TST reduced sequence length', '8. TST multivariate']
#Legend = ['9. LSTM', '10. LSTM multivariate']

draw = ['ret_acc_1_100']

# 'cum_profit'              line chart of cumulative profit over time when investing 1$ every day
# 'avg_d_return_by_year'    bar chart of average daily return per year
# 'avg_accuracy_by_year'    bar chart of average accuracy per year
# 'distribution_d_return'   histogram of distribution of daily returns
# 'distribution_p_return'   histogram of distribution of portfolio returns
# 'industry share'          line chart of share of industries in trades stocks over time
# 'ret_std_sharpe per size' bar chart matrix of total average return, standard deviation and sharpe ratio of differing portfolio sizes
#        ! ! !              -> further parameter in loading section
# 'ret_acc_1_100'           grouped bar chart of return and accuracy for k=1 and 100


""" LOAD DATA """
timeline = np.loadtxt(f'resources/Timeline_010190_311220.csv', dtype='str', delimiter=',')
years = np.array(range(1990,2021))
if 'cum_profit' in draw:
    i=0
    cum_profit_distr = np.empty(len(EIDs),dtype=np.ndarray)
    for EID in EIDs:
        cum_profit_distr[i] = np.loadtxt(f'analysis/' + str(EID) + 'cum_profit_distr' + '.csv', delimiter=',')
        i += 1
    cum_profit_distr = np.concatenate(cum_profit_distr).reshape((len(cum_profit_distr), len(cum_profit_distr[0]), -1))

if 'avg_d_return_by_year' in draw:
    i=0
    avg_d_return_by_year = np.empty(len(EIDs),dtype=np.ndarray)
    for EID in EIDs:
        avg_d_return_by_year[i] = np.loadtxt(f'analysis/' + str(EID) + 'avg_d_return_by_year' + '.csv',delimiter=',')
        i += 1
    avg_d_return_by_year = np.concatenate(avg_d_return_by_year).reshape((len(avg_d_return_by_year), len(avg_d_return_by_year[0]), -1))

if 'avg_accuracy_by_year' in draw:
    i=0
    avg_accuracy_by_year = np.empty(len(EIDs),dtype=np.ndarray)
    for EID in EIDs:
        avg_accuracy_by_year[i] = np.loadtxt(f'analysis/' + str(EID) + 'avg_accuracy_by_year' + '.csv',delimiter=',')
        i += 1
    avg_accuracy_by_year = np.concatenate(avg_accuracy_by_year).reshape((len(avg_accuracy_by_year), len(avg_accuracy_by_year[0]), -1))

if 'distribution_d_return' in draw:
    i = 0
    d_return = np.empty(len(EIDs), dtype=np.ndarray)
    for EID in EIDs:
        d_return[i] = np.loadtxt(f'experiment/return_' + str(EID) + '.csv', delimiter=',')[0:8088]
        i += 1
    d_return = np.concatenate(d_return).reshape((len(d_return), len(d_return[0]), -1))

if 'distribution_p_return' in draw:
    i = 0
    p_returns = np.empty(len(EIDs), dtype=np.ndarray)
    for EID in EIDs:
        d_return = np.loadtxt(f'experiment/return_' + str(EID) + '.csv', delimiter=',')[0:8088]
        first = np.where(d_return[:, 0] != 0)[0][0]
        p_returns[i] = np.nanmean(d_return[first:,1])
        i += 1
    monkey_p_returns = np.loadtxt(f'experiment/Monkey_Portfolioreturns2000' + '.csv', delimiter=',')
    SP500_returns = np.loadtxt(f'analysis/SP500_stocks_long_return' + '.csv', delimiter=',')

if 'industry share' in draw:
    i = 0
    industry_proportion_by_year = np.empty(len(EIDs), dtype=np.ndarray)
    for EID in EIDs:
        industry_proportion_by_year[i] = np.loadtxt(
            f'analysis/' + str(EID) + 'industry_proportion_by_year' + '.csv', delimiter=',')
        i += 1
    industry_proportion_by_year = np.concatenate(industry_proportion_by_year).reshape((len(industry_proportion_by_year), len(industry_proportion_by_year[0]), -1))
    industry_list = ['financials', 'utility', 'technology', 'ressources', 'consumer', 'other']

if 'ret_std_sharpe per size' in draw:
    import resoures.graphics_resources as gr

    i = 0
    total_avg_return = np.empty(len(EIDs), dtype=np.ndarray)
    std_d_return = np.empty(len(EIDs), dtype=np.ndarray)
    sharpe = np.empty(len(EIDs), dtype=np.ndarray)

    for EID in EIDs:
        total_avg_return[i] = np.loadtxt(f'analysis/' + str(EID) + 'total_avg_return' + '.csv',
                                         delimiter=',')
        std_d_return[i] = np.loadtxt(f'analysis/' + str(EID) + 'std_d_return' + '.csv', delimiter=',')
        sharpe[i] = np.loadtxt(f'analysis/' + str(EID) + 'sharpe' + '.csv', delimiter=',')
        i += 1
    total_avg_return = np.concatenate(total_avg_return).reshape((len(total_avg_return), len(total_avg_return[0]), -1))
    std_d_return = np.concatenate(std_d_return).reshape((len(std_d_return), len(std_d_return[0]), -1))
    sharpe = np.concatenate(sharpe).reshape((len(sharpe), len(sharpe[0]), -1))
    Portfolio_size = [3,1,5,10,15,20,25,30,35,40,45,50,75,100]
                    #[0,1,2,3, 4, 5, 6, 7, 8, 9, 10,11,12,13 ]

    #Portfolio sizes to draw: (select indices in regard to above order) + Redefine Portfolio_size
    idx = [1,0,3,7,13]
    Portfolio_size = [1,3,10,30,100]
    total_avg_return = total_avg_return[:,idx,:] * 100
    std_d_return = std_d_return[:,idx,:] * 100
    sharpe = sharpe[:,idx,:]

    if '2001' in EIDs:
        idx = np.where(np.array(EIDs) == '2001')
        for i in range(1,len(Portfolio_size)):
            total_avg_return[idx,i] = total_avg_return[idx,0]
            std_d_return[idx, i] = std_d_return[idx,0]
            sharpe[idx,i] = sharpe[idx, 0]

if 'ret_acc_1_100' in draw:
    import resources.graphics_resources as gr

    i = 0
    p_returns = np.empty(len(EIDs), dtype=np.ndarray)
    p_accuracy = np.empty(len(EIDs), dtype=np.ndarray)
    for EID in EIDs:
        d_return = np.loadtxt(f'experiment/return_' + str(EID) + '.csv', delimiter=',')[0:8088]
        d_accuracy = np.loadtxt(f'experiment/accuracy_' + str(EID) + '.csv', delimiter=',')[0:8088]
        first = np.where(d_return[:, 1] != 0)[0][0]
        p_returns[i] = np.array((np.nanmean(d_return[first:,1]),np.nanmean(d_return[first:,13])))
        p_accuracy[i] = np.nanmean(d_accuracy[first:, 0])
        i += 1
    p_returns = np.concatenate(p_returns).reshape((len(p_returns), len(p_returns[0]), -1))
    p_accuracy = p_accuracy.reshape((len(p_accuracy), 1))
    Portfolio_size = [1,100]


""" PLOT """

""" cumulative Profit """
if 'cum_profit' in draw:
    cum_profit = pd.DataFrame(cum_profit_distr[:, :, 1].transpose(1,0), index=timeline[1, 1:],
                              columns=Names)

    fig, ax = plt.subplots(figsize=(15, 4))
    chart = sns.lineplot(data=cum_profit, ax=ax)
    ax.xaxis.set_major_formatter(dates.DateFormatter("%Y"))
    #ax.set_xticklabels(pd.DataFrame(time_axis).iloc[::539])
    #ax.set_xticklabels(timeline[1,1:][range(0,8088,523)])
    ax.set_xticklabels([1990,1992,1994,1996,1998,2000,2002,2004,2006,2008,2010,2012, 2014, 2016, 2018, 2020])
    # set the xticks at the same frequency as the xlabels
    xtix = ax.get_xticks()
    ax.set_xticks(xtix[::539])
    fig.autofmt_xdate()

    plt.show()

""" avg_d_return_by_year """
if 'avg_d_return_by_year' in draw:

    avg_d_return = pd.DataFrame()
    for i in range(0,len(avg_d_return_by_year)):
        temp_df = pd.DataFrame(avg_d_return_by_year[i, :, 1],columns=['avg_d_return'])
        temp_df = temp_df.assign(year=years,model=Names[i])
        avg_d_return = pd.concat([avg_d_return, temp_df])

    fig, ax = plt.subplots(figsize=(15,4))
    chart = sns.catplot(data=avg_d_return, kind='bar',x='year',y='avg_d_return',hue='model', height=4, aspect=3.75, legend_out=False)
    chart.set_axis_labels('','')
    plt.show()

""" avg_accuracy_by_year """
if 'avg_accuracy_by_year' in draw:

    avg_accuracy = pd.DataFrame()
    for i in range(0, len(avg_accuracy_by_year)):
        temp_df = pd.DataFrame(avg_accuracy_by_year[i, :, 0], columns=['avg_accuracy'])
        temp_df = temp_df.assign(year=years, model=Names[i])
        avg_accuracy = pd.concat([avg_accuracy, temp_df])

    fig, ax = plt.subplots()

    chart = sns.catplot(data=avg_accuracy, kind='bar', x='year', y='avg_accuracy', hue='model', height=4, aspect=3.75, legend_out=False, facet_kws={'ylim':(0.49,0.545)})

    #ax = sns.barplot(data=avg_accuracy, x='year', y='avg_accuracy', hue='model', legend_out=False)
    ax.set(ylim=(0.375,0.55))
    chart.set_axis_labels('','')
    plt.show()

""" distribution_d_return """
if 'distribution_d_return' in draw:

    returns = pd.DataFrame()
    for i in range(0, len(d_return)):
        first = np.where(d_return[i,:,0]!=0)[0][0]
        temp_df = pd.DataFrame(d_return[i,first:,2], columns=['daily return'])
        temp_df = temp_df.assign(model=Names[i])
        returns = pd.concat([returns, temp_df])

    fig, ax = plt.subplots(figsize=(15,4))
    #ax = sns.histplot(data = returns , x = 'daily return' ,alpha = .7 ,hue = 'model', binwidth=0.1, kde=True, stat='probability')
    ax = sns.histplot(data = returns , x = 'daily return' ,alpha = .6 ,hue = 'model', binwidth=0.2, kde=True, stat='probability')
    ax.set(xlim=(-10,10))
    plt.show()

""" distribution_p_return """
if 'distribution_p_return' in draw:

    #returns = pd.DataFrame(p_returns, columns=['portfolio return'])
    #returns = returns.assign(model=Names)
    returns = pd.DataFrame(monkey_p_returns[:1000], columns=['average daily return'])
    returns = returns.assign(model='monkey')
    temp_df = pd.DataFrame(SP500_returns, columns=['average daily return'])
    temp_df = temp_df.assign(model='SP500 consituents long')
    returns = pd.concat([returns, temp_df])

    fig, ax = plt.subplots(figsize=(15,3))
    ax = sns.histplot(data = returns , x = 'average daily return' ,alpha = .6 ,hue = 'model', binwidth=0.005, kde=True, stat='probability', color=['blue', 'orange'])
    ax.set(xlim=(-0.15,0.4))
    for i in range(len(p_returns)):
        plt.axvline(p_returns[i], 0,1, color='green', linewidth=3)
        plt.text(p_returns[i], 0.85, Names[i], color='green')
    #monkey 99-percentiles
    plt.axvline(0.0372, 0,1, color='blue')
    plt.axvline(-0.0376, 0, 1, color='blue')
    #SP500_stock_long 99-percentiles
    plt.axvline(0.1891, 0, 1 ,color='orange')
    plt.axvline(-0.0321, 0, 1, color='orange')

    plt.show()

""" industry share """
if 'industry share' in draw:

    industry_proportion_by_2year = np.zeros((len(industry_proportion_by_year),int(len(industry_proportion_by_year[0])/2)+1,len(industry_proportion_by_year[0,0])))
    for i in range(15):
        industry_proportion_by_2year[:,i,:]=np.sum(industry_proportion_by_year[:,i*2:i*2+1,:],axis=1)
    industry_proportion_by_2year[:,-1,:]=industry_proportion_by_year[:,-1,:]
    industry_proportion_by_year = industry_proportion_by_2year
    years = [1990,1992,1994,1996,1998,2002,2002,2004,2006,2008,2010,2012,2014,2016,2018,2020]


    industry_per_year = pd.DataFrame()
    for i in range(0, len(industry_proportion_by_year)):
        for j in range(len(industry_list)):
            temp_df = pd.DataFrame(industry_proportion_by_year[i, :, j], columns=['frequency'])
            temp_df = temp_df.assign(year=years, model=Names[i], industry=industry_list[j])
            industry_per_year = pd.concat([industry_per_year, temp_df])

    fig, ax = plt.subplots(figsize=(15,6))
    chart = sns.lineplot(data=industry_per_year, x='year',y='frequency',hue='industry' )

    plt.show()

""" ret_std_shapre per size """
if 'ret_std_sharpe per size' in draw:

    returns = pd.DataFrame()
    for i in range(0, len(total_avg_return)):
        for j in range(len(Portfolio_size)):
            temp_df = pd.DataFrame(total_avg_return[i, j, :] , columns=['value'])
            temp_df = temp_df.assign(model=Names[i], k=Portfolio_size[j], metric='Return [%]')
            returns = pd.concat([returns, temp_df])
    st_deviation = pd.DataFrame()
    for i in range(0, len(std_d_return)):
        for j in range(len(Portfolio_size)):
            temp_df = pd.DataFrame(std_d_return[i, j,:], columns=['value'])
            temp_df = temp_df.assign(model=Names[i], k=Portfolio_size[j], metric='Standard deviation[%]')
            st_deviation = pd.concat([st_deviation, temp_df])
    sharpe_r = pd.DataFrame()
    for i in range(0, len(sharpe)):
        for j in range(len(Portfolio_size)):
            temp_df = pd.DataFrame(sharpe[i, j,:], columns=['value'])
            temp_df = temp_df.assign(model=Names[i], k=Portfolio_size[j], metric='Sharpe ratio')
            sharpe_r = pd.concat([sharpe_r, temp_df])
    data = pd.concat([returns, st_deviation, sharpe_r])

    #fig, ax = plt.subplots(figsize=(15,4))
    chart = sns.FacetGrid(data, col='k', row='metric', sharey='row', sharex=False, margin_titles=True, height=2.6, aspect=1.6)
    chart.map(sns.barplot, 'model', 'value')
    gr.show_values(chart, orientation='vertical')
    chart.set_axis_labels('','')
    plt.show()

""" ret_acc_1_100 """
if 'ret_acc_1_100' in draw:

    performance = pd.DataFrame()
    for i in range(0, len(p_returns)):
        for j in range(0,2):
            temp_df = pd.DataFrame(p_returns[i,j], columns=['value'])
            temp_df = temp_df.assign(model=Names[i], k=Portfolio_size[j], metric='Return [%]', Metric=f'Return , k=2*'+str(Portfolio_size[j])+' in %')
            performance = pd.concat([performance, temp_df])
    for i in range(0, len(p_accuracy)):
        temp_df = pd.DataFrame(p_accuracy[i], columns=['value'])
        temp_df =  temp_df.assign(model=Names[i], k=530, metric='Accuracy', Metric='Accuracy')
        performance = pd.concat([performance, temp_df])

    '''performance = performance.assign(architecture=['TST','TST','TST','TST','TST','TST','TST','TST','TST','TST',
                                          'TST','TST','TST','TST','LSTM','LSTM','LSTM','LSTM','TST',
                                          'TST','TST','TST','TST','TST','TST','LSTM','LSTM'])'''

    #chart = sns.FacetGrid(performance, col='Metric', row='architecture', sharey=False, sharex=False, margin_titles=True)#, height=4, aspect=0.6
    chart = sns.FacetGrid(performance, col='Metric', sharey=False, sharex=False, margin_titles=True,height=len(Names)*0.5+1, aspect=10/(len(Names)+2))  # , height=4, aspect=0.6
    chart.map(sns.barplot, 'value', 'model')

    chart.axes[0,0].set_xlim(0.15,0.4)
   #chart.axes[1,0].set_xlim(0.15,0.4)
    chart.axes[0,1].set_xlim(0.01,0.1)
   #chart.axes[1,1].set_xlim(0.01,0.1)
    chart.axes[0,2].set_xlim(0.49,0.53)
   #chart.axes[1,2].set_xlim(0.49,0.53)
    gr.show_values(chart,orientation='horizontal')

    plt.show()


print('finish')