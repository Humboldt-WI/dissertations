# Model from https://github.com/360jinrong/GBST
# Authors: Miaojun Bai, Yan Zheng, and Yun Shen
# From paper Gradient Boosting Survival Tree with Applications in Credit Scoring
# Code was modified to fit in the package

import numpy as np
from gbst.sklearn import gbstModel
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from DeepCreditSurv.datasets import load_M1
from DeepCreditSurv.datasets import load_M2
from DeepCreditSurv.datasets import load_ppdai
from DeepCreditSurv.datasets import load_LendingClub
from sksurv.metrics import concordance_index_ipcw

class gbst:
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
            self.x, self.y = self._make_data(_col_to_norm, _cat_feats, False)

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
            self.x, self.y = self._make_data(_col_to_norm, _cat_feats, True)
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
            _cat_feats = ['Gender', 'BorrowerRating', 'LoanType']
            self.x, self.y = self._make_data(_col_to_norm, _cat_feats, True)
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
            _cat_feats = ['grade', 'emp_length', 'home_ownership',
                          'verification_status']
            self.x, self.y = self._make_data(_col_to_norm, _cat_feats, True)
        else:
            raise ValueError(dataset + "is not a valid dataset, please give a valid dataset name!")

    def _make_data(self, col_to_std, cat_feats, to_dummy=False):
        x1 = self.df[col_to_std]
        x1 = StandardScaler().fit_transform(x1)
        if to_dummy:
            x2 = pd.get_dummies(self.df[cat_feats])
        else:
            x2 = self.df[cat_feats]
        x = np.concatenate([x1, x2], axis=1)
        y = self.df[[self.event_col_name, self.time_col_name]].to_records(index=False)
        return x, y

    def build_model_grid_cv(self):
        train_x, test_x, train_y, test_y = train_test_split(self.x, self.y, test_size=0.2, random_state=2)
        num_class = self.df[self.time_col_name].max() + 1
        classifier_cv = gbstModel(num_class=num_class)
        param_grid = {"n_estimators": range(40, 140, 10), "max_depth": range(3, 5, 1)}
        GridCV = GridSearchCV(classifier_cv, param_grid, cv=None, verbose=3)
        GridCV.fit(train_x, train_y[self.time_col_name])
        print(GridCV.best_params_)
        print(GridCV.best_score_)
        return GridCV.best_params_, GridCV.best_score_

    def build_model(self, params, evaluation=True):
        train_x, test_x, train_y, test_y = train_test_split(self.x, self.y, test_size=0.2, random_state=2)
        num_class = self.df[self.time_col_name].max() + 1
        classifier = gbstModel(n_estimators=params['n_estimators'],
                               max_depth=params['max_depth'], num_class=num_class)
        classifier.fit(X=train_x, y=train_y[self.time_col_name],
                       eval_set=[(train_x, train_y[self.time_col_name]), (test_x, test_y[self.time_col_name])],
                       verbose=True)
        hazards = classifier.predict_hazard(data=test_x)
        if evaluation:
            self.model_eval(hazards, train_y, test_y)

        return hazards

    def model_eval(self, hazards, train_y, test_y):
        train_y = train_y.astype([('e', bool), ('t', float)])
        test_y = test_y.astype([('e', bool), ('t', float)])
        horizons = [0.25, 0.5, 0.75, 1]
        times = np.quantile(train_y['t'][train_y['e'] == 1], horizons).tolist()
        cis = []
        for i, _ in enumerate(times):
            cis.append(concordance_index_ipcw(train_y, test_y, hazards[:, i], times[i])[0])

        for horizon in enumerate(horizons):
            print(f"For {horizon[1]} quantile,")
            print("TD Concordance Index:", cis[horizon[0]])
        return cis
