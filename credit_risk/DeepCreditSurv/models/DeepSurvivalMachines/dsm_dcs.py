import pandas as pd
import numpy as np
from DeepCreditSurv.datasets import load_M1
from DeepCreditSurv.datasets import load_M2
from DeepCreditSurv.datasets import load_ppdai
from DeepCreditSurv.datasets import load_LendingClub
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid
from DeepCreditSurv.models.DeepSurvivalMachines.dsm import DeepSurvivalMachines
from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc


class DSM:
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
            self.x, self.t, self.e = self._make_data(_col_to_norm, _cat_feats, False)

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
            self.x, self.t, self.e = self._make_data(_col_to_norm, _cat_feats, False)

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
            self.x, self.t, self.e = self._make_data(_col_to_norm, _cat_feats, True)

        elif dataset == 'LC':
            lc = load_LendingClub.LendingClub(file_path)
            self.df = lc.load_data()
            self.df = self.df[:100000]
            self.time_col_name = "time"
            self.event_col_name = "default_ind"
            _col_to_norm = ['loan_amnt', 'funded_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
                            'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                            'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
                            'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                            'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
                            'collection_recovery_fee', 'last_pymnt_amnt', 'collections_12_mths_ex_med',
                            'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']
            _cat_feats = ['grade',  'home_ownership',
                          'verification_status']
            self.x, self.t, self.e = self._make_data(_col_to_norm, _cat_feats, True)
        else:
            raise ValueError(dataset + "is not a valid dataset, please give a valid dataset name!")

    def _make_data(self, _col_to_std, _cat_feats, to_dummy=False):
        x1 = self.df[_col_to_std]
        x1 = MinMaxScaler().fit_transform(x1)
        if to_dummy:
            x2 = pd.get_dummies(self.df[_cat_feats])
        else:
            x2 = self.df[_cat_feats]
        x = np.concatenate([x1, x2], axis=1)
        x = StandardScaler().fit_transform(x)
        t = self.df[self.time_col_name].values
        e = self.df[self.event_col_name].values
        remove = ~np.isnan(t)
        return x[remove], t[remove], e[remove]

    def build_model(self):
        horizons = [0.25, 0.5, 0.75]
        times = np.quantile(self.t[self.e == 1], horizons).tolist()
        n = len(self.x)

        tr_size = int(n * 0.70)
        vl_size = int(n * 0.10)
        te_size = int(n * 0.20)

        x_train, x_test, x_val = self.x[:tr_size], self.x[-te_size:], self.x[tr_size:tr_size + vl_size]
        t_train, t_test, t_val = self.t[:tr_size], self.t[-te_size:], self.t[tr_size:tr_size + vl_size]
        e_train, e_test, e_val = self.e[:tr_size], self.e[-te_size:], self.e[tr_size:tr_size + vl_size]

        param_grid = {'k': [3, 4, 6],
                      'distribution': ['LogNormal', 'Weibull'],
                      'learning_rate': [1e-4, 1e-3],
                      'layers': [[100], [100, 100]]
                      }
        params = ParameterGrid(param_grid)

        models = []
        for param in params:
            model = DeepSurvivalMachines(k=param['k'],
                                         distribution=param['distribution'],
                                         layers=param['layers'])
            # The fit method is called to train the model
            model.fit(x_train, t_train, e_train, iters=100, learning_rate=param['learning_rate'])
            models.append([[model.compute_nll(x_val, t_val, e_val), model]])
        best_model = min(models)
        print(best_model[0][1])
        model = best_model[0][1]

        out_risk = model.predict_risk(x_test, times)
        out_survival = model.predict_survival(x_test, times)

        cis = []
        brs = []

        et_train = np.array([(e_train[i], t_train[i]) for i in range(len(e_train))],
                            dtype=[('e', bool), ('t', float)])
        et_test = np.array([(e_test[i], t_test[i]) for i in range(len(e_test))],
                           dtype=[('e', bool), ('t', float)])
        et_val = np.array([(e_val[i], t_val[i]) for i in range(len(e_val))],
                          dtype=[('e', bool), ('t', float)])


        for i, _ in enumerate(times):
            cis.append(concordance_index_ipcw(et_train, et_test, out_risk[:, i], times[i])[0])
        brs.append(brier_score(et_train, et_test, out_survival, times)[1])
        roc_auc = []
        for i, _ in enumerate(times):
            roc_auc.append(cumulative_dynamic_auc(et_train, et_test, out_risk[:, i], times[i])[0])
        for horizon in enumerate(horizons):
            print(f"For {horizon[1]} quantile,")
            print("TD Concordance Index:", cis[horizon[0]])
            print("Brier Score:", brs[0][horizon[0]])
            print("ROC AUC ", roc_auc[horizon[0]][0], "\n")

        return model, cis, brs
