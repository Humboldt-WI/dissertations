import pandas as pd

import random
from DeepCreditSurv.datasets import load_M1
from DeepCreditSurv.datasets import load_M2
from DeepCreditSurv.datasets import load_ppdai
from DeepCreditSurv.datasets import load_LendingClub
from DeepCreditSurv.models.DATE.models import DATE
from DeepCreditSurv.models.DATE.models import DATE_AE
from DeepCreditSurv.models.DATE.preprocessing import one_hot_encoder, one_hot_indices
from DeepCreditSurv.models.DATE.utilities import generate_data

random.seed(1234)


class DATE_EXP:
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
            encoded_indices = []
            for col in _cat_feats:
                idx_temp = []
                idx_temp.append(self.df.columns.get_loc(col))
                encoded_indices.append(idx_temp)

            self.preprocessed = generate_data(self.df, self.time_col_name, self.event_col_name,
                                              _col_to_norm, encoded_indices)

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
            self.df = one_hot_encoder(self.df, encode=_cat_feats)
            encoded_indices = one_hot_indices(self.df, _cat_feats)
            self.preprocessed = generate_data(self.df, self.time_col_name, self.event_col_name,
                                              _col_to_norm, encoded_indices)

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
            self.df = one_hot_encoder(self.df, encode=_cat_feats)
            encoded_indices = one_hot_indices(self.df, _cat_feats)
            self.preprocessed = generate_data(self.df, self.time_col_name, self.event_col_name,
                                              _col_to_norm, encoded_indices)

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
            self.df = one_hot_encoder(self.df, encode=_cat_feats)
            encoded_indices = one_hot_indices(self.df, _cat_feats)
            self.preprocessed = generate_data(self.df, self.time_col_name, self.event_col_name,
                                              _col_to_norm, encoded_indices)
        else:
            raise ValueError(dataset + "is not a valid dataset, please give a valid dataset name!")

    def build_model(self):
        r_epochs = 200

        # Two date models to choose
        simple = True
        if simple:
            model = DATE
        else:
            model = DATE_AE

        data_set = self.preprocessed
        train_data, valid_data, test_data, end_t, covariates, one_hot_indices, imputation_values \
            = data_set['train'], \
              data_set['valid'], \
              data_set['test'], \
              data_set['end_t'], \
              data_set['covariates'], \
              data_set[
                  'one_hot_indices'], \
              data_set[
                  'imputation_values']

        print("imputation_values:{}, one_hot_indices:{}".format(imputation_values, one_hot_indices))
        print("end_t:{}".format(end_t))
        train = {'x': train_data['x'], 'e': train_data['e'], 't': train_data['t']}
        valid = {'x': valid_data['x'], 'e': valid_data['e'], 't': valid_data['t']}
        test = {'x': test_data['x'], 'e': test_data['e'], 't': test_data['t']}

        perfomance_record = []

        date = model(batch_size=350,
                     learning_rate=3e-4,
                     beta1=0.9,
                     beta2=0.999,
                     require_improvement=1000,
                     num_iterations=10000, seed=31415,
                     l2_reg=0.001,
                     hidden_dim=[50, 50],
                     train_data=train, test_data=test, valid_data=valid,
                     input_dim=train['x'].shape[1],
                     num_examples=train['x'].shape[0], keep_prob=0.8,
                     latent_dim=50, end_t=end_t,
                     path_large_data='D:\\DeepCreditSurv\\DeepCreditSurv\\models\\DATE',
                     covariates=covariates,
                     categorical_indices=one_hot_indices,
                     disc_updates=1,
                     sample_size=200, imputation_values=imputation_values,
                     max_epochs=r_epochs, gen_updates=2)

        with date.session:
            date.train_test()
