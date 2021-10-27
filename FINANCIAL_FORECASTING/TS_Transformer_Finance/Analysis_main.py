import numpy as np
import scipy.stats as sc
import math
from TS_Transformer_Finance.resources import analysis_resources as ar

import warnings
warnings.filterwarnings('ignore')
print("No Warning Shown")



"""________PARAMETERS________"""
EID = '9'
# Experiment ID (str) of results files to analyse.
# EIDs for provided data refers to experiment ID from thesis paper
# 'Transformer Architecture in Deep Learning for Financial Price Forecasting'

trades_available = False
# Boolean, whether to analyse trading choices.

#time_path = '/Users\Fabian\PycharmProjects\InitiumNovum\TS_Transformer_Finance\resources\Timeline_010190_311220.csv'
d_return = np.loadtxt(f'experiment/return_' + str(EID) + '.csv', delimiter=',')[0:8089] / 100 # 1=100%
d_accuracy = np.loadtxt(f'experiment/accuracy_' + str(EID) + '.csv', delimiter=',')[0:8089]
timeline = np.loadtxt(f'resources\Timeline_010190_311220.csv', dtype='str', delimiter=',')
# File location of resource and experiment data.

n_stocks = 531
n_total_days, _ = d_return.shape
for day in range(n_total_days):
    if d_return.sum(axis=1)[day] != 0:
        first_pred_day = day
        break
for day in range(n_total_days-1, first_pred_day, -1):
    if d_return.sum(axis=1)[day] != 0:
        last_pred_day = day
        break
n_pred_days = last_pred_day - first_pred_day + 1

""" --- Performance Measures --- """
# Average return over the complete timespan (combined, long, short) in %
total_avg_return = d_return[first_pred_day:last_pred_day+1].mean(axis=0) # result verified
# Average accuracy over the complete timespan (acc,pTtT,pFtF,pTtF,pFtT)
total_avg_accuracy = d_accuracy[first_pred_day:last_pred_day+1].mean(axis=0) # resutl verified
print('  ---     PERFORMANCE MEASURES      ---',
      '\n Mean Accuracy:              ',np.round_(total_avg_accuracy[0], 5),
      '\n Trades (long/short each)       :  3,     1,      5,     10,    100',
      '\n Mean Return:               ',np.round_(total_avg_return[[0,1,2,3,13]], 5),
      '\n Mean Return (long):        ',np.round_(total_avg_return[[14,15,16,17,27]], 5),
      '\n Mean Return (short):       ',np.round_(total_avg_return[[28,29,30,31,41]],5))

# Standard Error
st_error = np.std(d_return[first_pred_day:last_pred_day+1], axis=0, ddof=1) / np.sqrt(n_pred_days)

# t-statistic
t_stat, p_value = sc.stats.ttest_1samp(d_return[first_pred_day:last_pred_day+1], popmean=0, axis=0, alternative='greater')
critical_value = sc.t.ppf(q=1-.05/2,df=len(d_return[first_pred_day:last_pred_day+1]))

# t-statistic accuracy
t_stat_acc, p_value_acc = sc.stats.ttest_1samp(d_accuracy[first_pred_day:last_pred_day+1], popmean=0.5, axis=0, alternative='greater')
critical_value_acc = sc.t.ppf(q=1-.05/2,df=len(d_accuracy[first_pred_day:last_pred_day+1]))

print(' Standard error:            ',np.round_(st_error[[0,1,2,3,13]], 5),
      '\n t-Statistic:               ',np.round_(t_stat[[0,1,2,3,13]], 5))
   #   '\n p-Value:                   ',np.round_(p_value[[0,1,2,3,13]],5),
   #   '\n critival_value:            ',np.round_(critical_value, 5))


#Minimum daily return in %
min_d_return = np.quantile(d_return[first_pred_day:last_pred_day+1], 0, axis=0) # result verified
#Quartile 1 daily return in %
Q1_d_return = np.quantile(d_return[first_pred_day:last_pred_day+1], 0.25, axis=0) # result verified
#Median daily return in %
median_d_return = np.quantile(d_return[first_pred_day:last_pred_day+1], 0.5, axis=0) # result verified
#quartile 3 daily return in %
Q3_d_return = np.quantile(d_return[first_pred_day:last_pred_day+1], 0.75, axis=0) # result verified
#Maximum daily return in %
max_d_return = np.quantile(d_return[first_pred_day:last_pred_day+1], 1, axis=0) # result verified
print(' Minimum:                   ',np.round_(min_d_return[[0,1,2,3,13]], 5),
      '\n Quartile 1:                ',np.round_(Q1_d_return[[0,1,2,3,13]], 5),
      '\n Median:                    ',np.round_(median_d_return[[0,1,2,3,13]], 5),
      '\n Quartile 3:                ',np.round_(Q3_d_return[[0,1,2,3,13]], 5),
      '\n Maximum:                   ',np.round_(max_d_return[[0,1,2,3,13]], 5))

#proportion of positive return days (profit)
temp_return = d_return[first_pred_day:last_pred_day+1].copy()
temp_return[d_return[first_pred_day:last_pred_day+1]<=0] = 0
pos_d_return = np.count_nonzero(temp_return, axis=0) / n_pred_days # result verified
#proportion of zero-profit days
temp_return = d_return[first_pred_day:last_pred_day+1].copy()
temp_return[d_return[first_pred_day:last_pred_day+1]==0] = 1
temp_return[d_return[first_pred_day:last_pred_day+1]!=0] = 0
zero_d_return = np.count_nonzero(temp_return, axis=0) / n_pred_days # result verified
#proportion of negative return days (loss)
temp_return = d_return[first_pred_day:last_pred_day+1].copy()
temp_return[d_return[first_pred_day:last_pred_day+1]>=0] = 0
neg_d_return = np.count_nonzero(temp_return, axis=0) / n_pred_days # result verified
print(' Share: ret > 0:            ',np.round_(pos_d_return[[0,1,2,3,13]], 5),
      '\n Share: ret = 0:            ',np.round_(zero_d_return[[0,1,2,3,13]], 5),
      '\n Share: ret < 0:            ',np.round_(neg_d_return[[0,1,2,3,13]], 5))

#Standard deviation of d_returns
std_d_return = np.std(d_return[first_pred_day:last_pred_day+1], axis=0) # result verified
#skewedness of d_returns
skew_d_return = sc.skew(d_return[first_pred_day:last_pred_day+1],axis=0) # result validated
#kurtosis of d_return (Fisher's definition, mean = 0)
kurt_d_return = sc.kurtosis(d_return[first_pred_day:last_pred_day+1], axis=0)
print(' Standard dev.:             ', np.round_(std_d_return[[0,1,2,3,13]], 5),
      '\n Skewness:                  ', np.round_(skew_d_return[[0,1,2,3,13]], 5),
      '\n Kurtosis:                  ', np.round_(kurt_d_return[[0,1,2,3,13]], 5))

#stard error of d_returns
#ste_d_return = smf.ols('a ~ 1 + b',data=d_return[first_pred_day:last_pred_day+1]).fit(cov_type='HAC',cov_kwds={'maxlags':1})
#print(ste_d_return.summary)

#VaR 1% (Gaussian)
z_score_1 = sc.norm.ppf(0.01)
VaR_1_gauss = - (total_avg_return + z_score_1 * std_d_return)
#CVaR 1% (Gaussian)
Z_score_1 = -(1/0.01) * (1/math.sqrt(2*math.pi)) * math.exp(-0.5 * math.pow(z_score_1,2))
CVaR_1_gauss = - (total_avg_return + Z_score_1 * std_d_return)
#VaR 5% (Gaussian)
z_score_5 = sc.norm.ppf(0.05)
VaR_5_gauss = - (total_avg_return + z_score_5 * std_d_return)
#CVaR 5% (Gaussian)
Z_score_5 = -(1/0.05) * (1/math.sqrt(2*math.pi)) * math.exp(-0.5 * math.pow(z_score_5,2))
CVaR_5_gauss = - (total_avg_return + Z_score_5 * std_d_return)

print('  ---       RISK CHARACTERISTICS       ---',
      '\n VaR 1%:                    ',np.round_(VaR_1_gauss[[0,1,2,3,13]], 5),
      '\n CVaR 1%:                   ',np.round_(CVaR_1_gauss[[0,1,2,3,13]], 5),
      '\n VaR 5%:                    ',np.round_(VaR_5_gauss[[0,1,2,3,13]],5),
      '\n CVaR 5%:                   ',np.round_(CVaR_5_gauss[[0,1,2,3,13]],5))

#maximum drawdown not yet fully correct
d_accr_gain = np.zeros_like(d_return)
d_accr_gain[0] = d_return[0]
for day in range(1,len(d_return)):
    d_accr_gain[day] = d_accr_gain[day - 1] + d_return[day]
max_drawdown = np.zeros(len(d_accr_gain[0]))
for day in range(1,n_pred_days):
    peak = d_accr_gain[first_pred_day:first_pred_day + day + 1].max(axis=0)
    drawdown = (d_accr_gain[first_pred_day + day] - peak) / peak
    for portf in range(len(d_accr_gain[0])):
        if drawdown[portf] < max_drawdown[portf]:
            max_drawdown[portf] = drawdown[portf]

#maximum drawdown with 50 days buildup
max_drawdown_buildup = np.zeros(len(d_accr_gain[0]))
for day in range(51,n_pred_days):
    peak = d_accr_gain[first_pred_day:first_pred_day + day + 1].max(axis=0)
    drawdown = (d_accr_gain[first_pred_day + day] - peak) / peak
    for portf in range(len(d_accr_gain[0])):
        if drawdown[portf] < max_drawdown_buildup[portf]:
            max_drawdown_buildup[portf] = drawdown[portf]

print(' Max. drawdown:             ',np.round_(max_drawdown[[0,1,2,3,13]], 5),
      '\n Max. drawdown adj.:        ', np.round_(max_drawdown_buildup[[0,1,2,3,13]], 5))

""" ANNUALIZED RISK-RETURN METRICS """
# Return p.a. (non-cumulative interest)
date_first_trade = np.datetime64(timeline[1,first_pred_day])
date_last_trade = np.datetime64(timeline[1,last_pred_day])

calendar_days = date_last_trade - date_first_trade
n_pred_years = calendar_days.astype('long') / 365.25
n_pred_days_per_year = np.round_(n_pred_days / n_pred_years)
return_pa_distr = total_avg_return * n_pred_days_per_year
return_pa_thes = np.power((1 + total_avg_return), n_pred_days_per_year)

# Excess Return
# ~Rf: 1 year historical US Treasury notes
return_treasury = np.array((0.0792, 0.0664, 0.0415, 0.0350, 0.0354, 0.0705, 0.0509, 0.0561, 0.0524, 0.0451,
                            0.0612, 0.0481, 0.0216, 0.0136, 0.0124, 0.0286, 0.0445, 0.0506, 0.0271, 0.0044,
                            0.0035, 0.0027, 0.0012, 0.0015, 0.0012, 0.0020, 0.0054, 0.0083, 0.0180, 0.0153, 0.0010, 0.0008 ))
#from https://www.multpl.com/1-year-treasury-rate/table/by-year (retrieved: 04.08.2021)

# ~Rm: daily 'market' return of the underlying stock index in the dataset
d_return_stockindex = ar.get_d_return_idx('load') #load or calculate
# Rm:
Rm = d_return_stockindex[first_pred_day:last_pred_day+1]
Rf = np.zeros_like(Rm)
year = np.vstack(np.core.defchararray.split(timeline[1,first_pred_day:last_pred_day+1], sep='-'))[:,0].astype(int)
for i in range(32):
    Rf[year == 1990 + i] = return_treasury[i]
d_Rf = np.power((1 + Rf), (1/n_pred_days_per_year)) - 1
# beta (trading_strategy to full index return)
covariance = np.zeros(len(d_return[0]))
for i in range(len(d_return[0])):
    covariance[i] = np.cov(Rm, d_return[first_pred_day:last_pred_day+1,i])[0,1]
variance = np.var(d_return[first_pred_day:last_pred_day+1])
beta = covariance / variance
# Expected return [ExpR = Rf + (beta * (Rm - Rf)), CAPM]
exp_d_return = np.stack(((d_Rf) for _ in range(42)), axis=1) + (beta * np.stack(((Rm - d_Rf) for _ in range(42)), axis=1))
# Expected return p.a.
exp_return_pa_distr = exp_d_return.mean(axis=0) * n_pred_days_per_year
exp_return_pa_thes = np.power((1 + exp_d_return.mean(axis=0)), n_pred_days_per_year)
# Excess Return [ExcR = d_return - ExpR): daily, p.a. dist, and p.a. thes
excess_d_return = d_return[first_pred_day:last_pred_day+1] - exp_d_return
excess_return_pa_distr = return_pa_distr - exp_return_pa_distr
excess_return_pa_thes = return_pa_thes - exp_return_pa_thes

print('  --- ANNUALIZED RISK-RETURN CHARACTERISTICS ---',
      '\n Return p.a. (distr):       ',np.round_(return_pa_distr[[0,1,2,3,13]], 5),
      '\n Return p.a. (thes):        ',np.round_(return_pa_thes[[0,1,2,3,13]], 5),
      '\n Excess return p.a. (distr):',np.round_(excess_return_pa_distr[[0,1,2,3,13]],5),
      '\n Excess return p.a. (thes): ',np.round_(excess_return_pa_thes[[0,1,2,3,13]],5))

# Annualized Standard Deviation
std_return_pa = std_d_return * np.sqrt(n_pred_days_per_year)
# Downside deviation
temp_return = d_return[first_pred_day:last_pred_day+1].copy()
temp_return[d_return[first_pred_day:last_pred_day+1]>=np.stack((d_Rf for _ in range(42)), axis=1)] = 0
downside_dev_d_return = np.sqrt(np.power(temp_return[first_pred_day:last_pred_day+1],2).mean(axis=0))
downside_dev_return_pa = downside_dev_d_return * np.sqrt(n_pred_days_per_year)

# Sharpe Ratio [sharpe =  (Rp - Rf) / std_p]
sharpe = (return_pa_distr - return_treasury.mean()) / std_return_pa
# Sortino Ratio [sortino = (Rp - Rf) / downside_dev
sortino = (return_pa_distr - Rf.mean()) / downside_dev_return_pa


print(' Standard dev. p.a.:        ', np.round_(std_return_pa[[0,1,2,3,13]], 5),
      '\n Downside dev.:             ',np.round_(downside_dev_d_return[[0,1,2,3,13]], 5),
      '\n Downside dev. p.a.:        ',np.round_(downside_dev_return_pa[[0,1,2,3,13]], 5),
      '\n Sharpe ratio p.a.:         ',np.round_(sharpe[[0,1,2,3,13]], 5),
      '\n Sortino ratio p.a.:        ',np.round_(sortino[[0,1,2,3,13]], 5))

"""     ANALYSIS OVER TIME     """
# Cumulative Profit (distr. and thes)
cum_profit_distr = np.zeros_like(d_return)
cum_profit_thes = np.zeros_like(d_return)
for i in range(1,n_total_days):
    cum_profit_distr[i] = cum_profit_distr[i - 1] + d_return[i]
    cum_profit_thes[i] = (1 + cum_profit_thes[i - 1]) * (1 + d_return[i]) - 1
# Thes. profits 'explode' unrealistically. >>> dropped

# Average daily returns by year
# (and Standard Deviation by year)
# and Sharpe Ratio by year
# and Average accuracy by year
year = np.vstack(np.core.defchararray.split(timeline[1,1:], sep='-'))[:,0].astype(int)
avg_d_return_by_year = np.zeros((31,42))
std_d_return_by_year = np.zeros((31,42))
sharpe_by_year = np.zeros((31,42))
avg_accuracy_by_year = np.zeros((31,5))
for i in range(31):
    avg_d_return_by_year[i] = d_return[year == 1990 + i].mean(axis=0)
    std_d_return_by_year[i] = np.std(d_return[year == 1990 + i], axis=0)
    days_in_year = np.count_nonzero(year[year == 1990 + i])
    sharpe_by_year[i] = (avg_d_return_by_year[i] * days_in_year - return_treasury[i]) / (std_d_return_by_year[i] * np.sqrt(days_in_year))
    avg_accuracy_by_year[i] = d_accuracy[year == 1990 + i].mean(axis=0)


""" TRADE ANALYSIS """
if trades_available == True:

    d_trades = np.loadtxt(f'experiment/trades_' + str(EID) + '.csv', delimiter=',')[0:8089]
    stock_list = np.loadtxt(f'resources/Constituents_Segments.csv', delimiter=',', dtype='str')
    industry_list = ['financials', 'utility', 'technology', 'ressources', 'consumer', 'other']
    industry_proportion = np.zeros(6)
    for i in range(6):
        industry_proportion[i] = np.count_nonzero(stock_list[:,1] == industry_list[i])/len(stock_list[:,0])


    d_industry = np.empty((len(d_trades),2), dtype=object)
    for i in range(len(d_trades)):
        if np.any(np.nonzero(d_trades[i] == 1)):
            d_industry[i,0] = str(stock_list[np.nonzero(d_trades[i] == 1),1][0][0])
        if np.any(np.nonzero(d_trades[i] == -1)):
            d_industry[i,1] = str(stock_list[np.nonzero(d_trades[i] == -1),1][0][0])

    industry_proportion_by_year = np.zeros((31,3,6))
    for i in range(31):
        for j in range(6):
            industry_proportion_by_year[i,0,j] = np.count_nonzero(d_industry[year == 1990 + i,0]==industry_list[j])
            industry_proportion_by_year[i,1,j] = np.count_nonzero(d_industry[year == 1990 + i,1]==industry_list[j])
            industry_proportion_by_year[i,2,j] = industry_proportion_by_year[i,0,j] + industry_proportion_by_year[i,1,j]


""" Data-Export """
if trades_available == True:
    np.savetxt(f'analysis/' + str(EID) + 'industry_proportion_by_year' + '.csv', industry_proportion_by_year[:,2,:], delimiter=',')
np.savetxt(f'analysis/' + str(EID) + 'cum_profit_distr' + '.csv', cum_profit_distr, delimiter=',')
np.savetxt(f'analysis/' + str(EID) + 'avg_d_return_by_year' + '.csv', avg_d_return_by_year, delimiter=',')
np.savetxt(f'analysis/' + str(EID) + 'std_d_return_by_year' + '.csv', std_d_return_by_year, delimiter=',')
np.savetxt(f'analysis/' + str(EID) + 'sharpe_by_year' + '.csv', sharpe_by_year, delimiter=',')
np.savetxt(f'analysis/' + str(EID) + 'avg_accuracy_by_year' + '.csv', avg_accuracy_by_year, delimiter=',')
np.savetxt(f'analysis/' + str(EID) + 'sharpe' + '.csv', sharpe, delimiter=',')
np.savetxt(f'analysis/' + str(EID) + 'total_avg_return' + '.csv', total_avg_return, delimiter=',')
np.savetxt(f'analysis/' + str(EID) + 'std_d_return' + '.csv', std_d_return, delimiter=',')


print('finished')



























