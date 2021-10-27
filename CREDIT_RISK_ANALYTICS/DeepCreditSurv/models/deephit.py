import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DeepCreditSurv.datasets import load_M1
from DeepCreditSurv.datasets import load_M2
from DeepCreditSurv.datasets import load_ppdai
from DeepCreditSurv.datasets import load_LendingClub

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch  # For building the networks
import torchtuples as tt  # Some useful functions

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

from sklearn.model_selection import GridSearchCV, train_test_split


class DeepHit:
    def __init__(self, dataset='M1', file_path=None):
        if dataset == 'M1':
            M1 = load_M1.M1(file_path)
            self.df = M1.load_data()
            self.time_col_name = "duration"
            self.event_col_name = "default_time"
            self.df = self.df.drop(["id", "first_time", "payoff_time", "status_time", "time"], axis=1)
            _col_to_norm = ['orig_time', 'mat_time', 'balance_time', 'LTV_time', 'interest_rate_time',
                            'hpi_time', 'gdp_time', 'uer_time', 'balance_orig_time',
                            'FICO_orig_time', 'LTV_orig_time', 'Interest_Rate_orig_time', 'hpi_orig_time']
            _cat_feats = ['REtype_CO_orig_time', 'REtype_PU_orig_time', 'REtype_SF_orig_time', 'investor_orig_time']
            self.x_train, self.x_val, self.x_test, self.labtrans, self.y_train, self.y_val, self.durations_test, self.events_test = self._make_data(
                _col_to_norm, _cat_feats, False)

        elif dataset == 'M2':
            M2 = load_M2.M2(file_path)
            self.df = M2.load_data()
            self.time_col_name = "time"
            self.event_col_name = "default"
            _col_to_norm = ['int.rate', 'orig.upb', 'fico.score', 'dti.r', 'ltv.r',
                            'bal.repaid', 'hpi.st.d.t.o', 'hpi.zip.o', 'hpi.zip.d.t.o',
                            'ppi.c.FRMA', 'TB10Y.d.t.o', 'FRMA30Y.d.t.o', 'ppi.o.FRMA',
                            'equity.est', 'hpi.st.log12m', 'hpi.r.st.us', 'hpi.r.zip.st',
                            'st.unemp.r12m', 'st.unemp.r3m', 'TB10Y.r12m', 'T10Y3MM', 'T10Y3MM.r12m']
            _cat_feats = ['t.act.12m', 't.del.30d.12m', 't.del.60d.12m']
            self.x_train, self.x_val, self.x_test, self.labtrans, self.y_train, self.y_val, self.durations_test, self.events_test = self._make_data(
                _col_to_norm, _cat_feats, True)

        elif dataset == 'ppdai':
            ppd = load_ppdai.ppdai(file_path)
            self.df = ppd.load_data()
            self.time_col_name = "Period"
            self.event_col_name = "RepaymentStatus"
            _col_to_norm = ['Loanstopay', 'interesttopay',
                            'RemainingPrinciple', 'RemainingInterest',
                            'LoanValue', 'LoanRate',
                            'Age', 'HistoryLoans', 'HistoryLoanValue',
                            'TotalLoanstopay', 'HistoryNormalrepaymonths', 'HistoryDefaultMonths'
                            ]
            _cat_feats = ['Gender', 'BorrowerRating', 'LoanType', 'LoanPeriod']
            self.x_train, self.x_val, self.x_test, self.labtrans, self.y_train, self.y_val, self.durations_test, self.events_test = self._make_data(
                _col_to_norm, _cat_feats, True)

        elif dataset == 'LC':
            lc = load_LendingClub.LendingClub(file_path)
            self.df = lc.load_data()
            self.time_col_name = "time"
            self.event_col_name = "default_ind"
            _col_to_norm = ['loan_amnt', 'funded_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
                            'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                            'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
                            'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                            'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
                            'collection_recovery_fee', 'last_pymnt_amnt', 'collections_12_mths_ex_med',
                            'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']
            _cat_feats = ['grade', 'emp_length', 'home_ownership', 'sub_grade', 'purpose',
                          'verification_status', 'term']
            self.x_train, self.x_val, self.x_test, self.labtrans, self.y_train, self.y_val, self.durations_test, self.events_test = self._make_data(
                _col_to_norm, _cat_feats, True)
        else:
            raise ValueError(dataset + "is not a valid dataset, please give a valid dataset name!")

    def _make_data(self, col_to_std, cat_feats, get_dummies = False):
        df = self.df
        if get_dummies:
            df = pd.get_dummies(df, columns=cat_feats)
            colname = df.columns
            cat_feats = colname.drop(self.time_col_name)
            cat_feats = cat_feats.drop(self.event_col_name)
            cat_feats = cat_feats.drop(col_to_std)
            print(cat_feats)
        df_train = df
        df_test = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.2)
        df_train = df_train.drop(df_val.index)

        standardize = [([col], StandardScaler()) for col in col_to_std]
        leave = [([col], None) for col in cat_feats]

        x_mapper = DataFrameMapper(standardize + leave)

        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_val = x_mapper.transform(df_val).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')
        labtrans, y_train, y_val, durations_test, events_test = self._label_transforms(df_train, df_test, df_val)

        return x_train, x_val, x_test, labtrans, y_train, y_val, durations_test, events_test

    def _label_transforms(self, df_train, df_test, df_val):
        num_durations = 60
        labtrans = DeepHitSingle.label_transform(num_durations)
        get_target = lambda df: (df[self.time_col_name].values, df[self.event_col_name].values)
        y_train = labtrans.fit_transform(*get_target(df_train))
        y_val = labtrans.transform(*get_target(df_val))

        # We don't need to transform the test labels
        durations_test, events_test = get_target(df_test)
        return labtrans, y_train, y_val, durations_test, events_test

    def find_lr(self):
        in_features = self.x_train.shape[1]
        num_nodes = [256, 256, 256, 256]
        out_features = self.labtrans.out_features
        batch_norm = True
        dropout = 0.2

        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

        model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=self.labtrans.cuts)

        batch_size = 256
        lr_finder = model.lr_finder(self.x_train, self.y_train, batch_size, tolerance=3)
        _ = lr_finder.plot()
        return lr_finder.get_best_lr()



    def build_model(self, lr=0.01):
        in_features = self.x_train.shape[1]
        num_nodes = [256, 256, 256, 256]
        out_features = self.labtrans.out_features
        batch_norm = True
        dropout = 0.2

        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

        val = (self.x_val, self.y_val)
        batch_size = 256
        model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=self.labtrans.cuts)
        model.optimizer.set_lr(lr)
        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]
        log = model.fit(self.x_train, self.y_train, batch_size, epochs, callbacks, val_data=val)
        _ = log.plot()
        surv = model.predict_surv_df(self.x_test)
        ev = EvalSurv(surv, self.durations_test, self.events_test, censor_surv='km')
        cis = ev.concordance_td('antolini')
        return cis
