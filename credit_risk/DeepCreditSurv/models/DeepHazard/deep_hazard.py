# Model from https://github.com/deniserava/DeepHazard
# Authors: Denise Rava, Jelena Bradic
# From paper deep_hazard: a Neural Network method for survival data for survival function estimation
# Code was modified to fit in the package
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import DeepCreditSurv.models.DeepHazard.ApplyDeepHaz as dh
from DeepCreditSurv.datasets import load_M1
from DeepCreditSurv.datasets import load_M2
from DeepCreditSurv.datasets import load_ppdai
from DeepCreditSurv.datasets import load_LendingClub
from sklearn.preprocessing import StandardScaler


class deep_hazard:
    def __init__(self, dataset='M1', file_path=None):
        if dataset == 'M1':
            M1 = load_M1.M1(file_path)
            self.df = M1.load_data()
            self.time_col_name = "duration"
            self.event_col_name = "default_time"
            self.df = self.df.drop(["id", "first_time", "payoff_time", "status_time", "time"], axis=1)
            _col_to_norm = ['orig_time', 'mat_time', 'balance_time', 'LTV_time',
                            'interest_rate_time', 'hpi_time', 'gdp_time', 'uer_time',
                            'balance_orig_time', 'FICO_orig_time', 'LTV_orig_time',
                            'Interest_Rate_orig_time', 'hpi_orig_time']
            self.df_standardization(_col_to_norm)
            self.df_normalization()


        elif dataset == 'M2':
            M2 = load_M2.M2(file_path)
            self.df = M2.load_data()
            self.time_col_name = "time"
            self.event_col_name = "default"
            self.df = self.df.drop(['label', 'payoff', 'current_year'], axis=1)
            _col_to_norm = ['int.rate', 'orig.upb', 'fico.score', 'dti.r', 'ltv.r',
                            'bal.repaid', 'hpi.st.d.t.o', 'hpi.zip.o', 'hpi.zip.d.t.o',
                            'ppi.c.FRMA', 'TB10Y.d.t.o', 'FRMA30Y.d.t.o', 'ppi.o.FRMA',
                            'equity.est', 'hpi.st.log12m', 'hpi.r.st.us', 'hpi.r.zip.st',
                            'st.unemp.r12m', 'st.unemp.r3m', 'TB10Y.r12m', 'T10Y3MM', 'T10Y3MM.r12m']
            _cat_feats = ['t.act.12m', 't.del.30d.12m', 't.del.60d.12m']

            self.df = pd.get_dummies(self.df, columns=_cat_feats)
            self.df_standardization(_col_to_norm)
            self.df_normalization()
        elif dataset == 'ppdai':
            ppd = load_ppdai.ppdai(file_path)
            self.df = ppd.load_data()
            self.time_col_name = "Period"
            self.event_col_name = "RepaymentStatus"
            _col_to_norm = ['Loanstopay', 'interesttopay',
                            'RemainingPrinciple', 'RemainingInterest',
                            'LoanValue', 'LoanPeriod', 'LoanRate',
                            'IsFirstTime', 'Age', 'PhoneVerified',
                            'RegistrationVerified', 'VideoVerified', 'DegreeVerified',
                            'CreditVerified', 'TaobaoVerified', 'HistoryLoans', 'HistoryLoanValue',
                            'TotalLoanstopay', 'HistoryNormalrepaymonths', 'HistoryDefaultMonths'
                            ]
            self.df = pd.get_dummies(self.df)
            self.df_standardization(_col_to_norm)
            self.df_normalization()

        elif dataset == 'LC':
            lc = load_LendingClub.LendingClub(file_path)
            self.df = lc.load_data()
            self.time_col_name = "time"
            self.event_col_name = "default_ind"
            _col_to_norm = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
                            'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                            'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
                            'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                            'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
                            'collection_recovery_fee', 'last_pymnt_amnt', 'collections_12_mths_ex_med',
                            'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']
            self.df = pd.get_dummies(self.df)
            self.df_standardization(_col_to_norm)
            self.df_normalization()
        else:
            raise ValueError(dataset + "is not a valid dataset, please give a valid dataset name!")

    def _create_train_test_set(self):
        df = self.df
        df = df[:30000]
        test = df.sample(frac=0.3)
        train = df.drop(test.index)
        return train, test

    def df_standardization(self, col_to_norm):
        x = self.df[col_to_norm].values
        x_scaled = StandardScaler().fit_transform(x)
        df_temp = pd.DataFrame(x_scaled, columns=col_to_norm, index=self.df.index)
        self.df[col_to_norm] = df_temp

    def df_normalization(self):
        t = self.df[self.time_col_name].values
        t = np.reshape(t, (-1, 1))
        t_scaled = MinMaxScaler().fit_transform(t)
        self.df[self.time_col_name] = t_scaled

    def build_model(self, l2c=1e-5, lrc=2e-1, structure=None,
                    init_method='he_uniform', optimizer='adam', num_epochs=100, early_stopping=1e-5,
                    penal='Ridge'):
        train, test = self._create_train_test_set()
        if not structure:
            structure = [{'activation': 'Relu', 'num_units': 10, 'dropout': 0.2},
                         {'activation': 'Relu', 'num_units': 10, 'dropout': 0.2}]

        deep_haz_lis, surv, c_index = dh.deephaz_const(train=train, test=test, l2c=l2c, lrc=lrc,
                                                       structure=structure, init_method=init_method,
                                                       optimizer=optimizer,
                                                       num_epochs=num_epochs, early_stopping=early_stopping,
                                                       penal=penal,
                                                       time_col_name=self.time_col_name,
                                                       event_col_name=self.event_col_name)
        return deep_haz_lis, surv, c_index
