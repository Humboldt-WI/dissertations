from DeepCreditSurv.models.DRSA.BASE_MODEL import BASE_RNN
import pandas as pd
import sys
import random
from DeepCreditSurv.datasets import load_M1
from DeepCreditSurv.datasets import load_M2
from DeepCreditSurv.datasets import load_ppdai
from DeepCreditSurv.datasets import load_LendingClub
from DeepCreditSurv import utils
from DeepCreditSurv.models.DRSA import feateng

random.seed(1234)


class DRSA:
    def __init__(self, dataset='M1', file_path=None, file_folder='D:\\DeepCreditSurv\\DeepCreditSurv\\datasets\\'):
        self.file_folder = file_folder
        self.dataset = dataset
        self.feat_index_f = self.file_folder + dataset + '\\feat_ind.txt'
        train_file = self.file_folder + dataset + '\\trainDRSA.txt'
        test_file = self.file_folder + dataset + '\\testDRSA.txt'
        self.yzbx_train = self.file_folder + dataset + '\\train_yzb.txt'
        self.yzbx_test = self.file_folder + dataset + '\\test_yzb.txt'
        if dataset == 'M1':
            M1 = load_M1.M1(file_path)
            self.df = M1.load_data()
            self.df['balance_time'] = self.df['balance_time'].astype(int)
            self.df['balance_orig_time'] = self.df['balance_orig_time'].astype(int)
            self.df = self.df.drop(["id", "first_time", "payoff_time", "status_time", "time"], axis=1)
            self.df.drop(self.df[self.df['balance_time'] > 1000000].index, inplace=True)
            self.df.drop(self.df[self.df['balance_orig_time'] > 1000000].index, inplace=True)
            self.features = ['orig_time', 'mat_time', 'balance_time', 'LTV_time',
                             'interest_rate_time', 'hpi_time', 'gdp_time', 'uer_time',
                             'REtype_CO_orig_time', 'REtype_PU_orig_time', 'REtype_SF_orig_time',
                             'investor_orig_time', 'balance_orig_time', 'FICO_orig_time',
                             'LTV_orig_time', 'Interest_Rate_orig_time', 'hpi_orig_time']
            self.time_col_name = "duration"
            self.event_col_name = "default_time"
            self.full_columns = self.features
            self.full_columns.append(self.time_col_name)
            self.full_columns.append(self.event_col_name)
            self.df = self.df[self.full_columns]
            self.df.head()
            df_test = self.df.sample(frac=0.2)
            df_train = self.df.drop(df_test.index)
            print("test:{}, train:{}, all: {}".format(len(df_test), len(df_train),
                                                      len(df_test) + len(df_train)))
            df_train.to_csv(train_file, index=False, header=False)
            df_test.to_csv(test_file, index=False, header=False)
            self.feat_index, self.feat_index_lines, self.multipliers = feateng.build_feat_index(self.feat_index_f,
                                                                                                self.features, self.df)
            feateng.build_yzbx_data(self.feat_index, train_file, test_file,
                                    self.yzbx_train,
                                    self.yzbx_test,
                                    self.features, self.multipliers)
            self.feature_size = 18
            self.max_den = max(self.feat_index.values())
            self.max_seq_len = int(118)
            print(self.max_den)

        elif dataset == 'M2':
            M2 = load_M2.M2(file_path)
            self.df = M2.load_data()
            self.df = self.df.drop(['label', 'current_year'], axis=True)
            self.features = ['int.rate', 'orig.upb', 'fico.score', 'dti.r', 'ltv.r', 'bal.repaid', 't.act.12m',
                             't.del.30d.12m', 't.del.60d.12m', 'hpi.st.d.t.o', 'hpi.zip.o',
                             'hpi.zip.d.t.o', 'ppi.c.FRMA', 'TB10Y.d.t.o', 'FRMA30Y.d.t.o',
                             'ppi.o.FRMA', 'equity.est', 'hpi.st.log12m', 'hpi.r.st.us',
                             'hpi.r.zip.st', 'st.unemp.r12m', 'st.unemp.r3m', 'TB10Y.r12m',
                             'T10Y3MM', 'T10Y3MM.r12m']
            self.time_col_name = "time"
            self.event_col_name = "default"
            self.full_columns = self.features
            self.full_columns.append(self.time_col_name)
            self.full_columns.append(self.event_col_name)
            self.df = self.df[self.full_columns]
            df_test = self.df.sample(frac=0.2)
            df_train = self.df.drop(df_test.index)
            print("test:{}, train:{}, all: {}".format(len(df_test), len(df_train),
                                                      len(df_test) + len(df_train)))
            df_train.to_csv(train_file, index=False, header=False)
            df_test.to_csv(test_file, index=False, header=False)
            self.feat_index, self.feat_index_lines, self.multipliers = feateng.build_feat_index(self.feat_index_f,
                                                                                                self.features, self.df)
            feateng.build_yzbx_data(self.feat_index, train_file, test_file,
                                    self.yzbx_train,
                                    self.yzbx_test,
                                    self.features, self.multipliers)
            self.feature_size = 26
            self.max_den = max(self.feat_index.values())
            self.max_seq_len = int(146)
            print(self.max_den)

        elif dataset == 'ppdai':
            ppd = load_ppdai.ppdai(file_path)
            self.df = ppd.load_data()
            self.df = pd.get_dummies(self.df)
            self.features = ['Loanstopay', 'interesttopay',
                             'RemainingPrinciple', 'RemainingInterest', 'LoanValue', 'LoanPeriod',
                             'LoanRate', 'IsFirstTime', 'Age', 'Gender', 'PhoneVerified',
                             'RegistrationVerified', 'VideoVerified', 'DegreeVerified',
                             'CreditVerified', 'TaobaoVerified', 'HistoryLoans', 'HistoryLoanValue',
                             'TotalLoanstopay', 'HistoryNormalrepaymonths', 'HistoryDefaultMonths',
                             'BorrowerRating_A', 'BorrowerRating_B', 'BorrowerRating_C',
                             'BorrowerRating_D', 'BorrowerRating_E', 'BorrowerRating_F',
                             'LoanType_E-commerce', 'LoanType_Normal', 'LoanType_Others']
            self.time_col_name = "Period"
            self.event_col_name = "RepaymentStatus"
            self.full_columns = self.features
            self.full_columns.append(self.time_col_name)
            self.full_columns.append(self.event_col_name)
            self.df = self.df[self.full_columns]
            df_test = self.df.sample(frac=0.2)
            df_train = self.df.drop(df_test.index)
            print("test:{}, train:{}, all: {}".format(len(df_test), len(df_train),
                                                      len(df_test) + len(df_train)))
            df_train.to_csv(train_file, index=False, header=False)
            df_test.to_csv(test_file, index=False, header=False)
            self.feat_index, self.feat_index_lines, self.multipliers = feateng.build_feat_index(self.feat_index_f,
                                                                                                self.features, self.df)
            feateng.build_yzbx_data(self.feat_index, train_file, test_file,
                                    self.yzbx_train,
                                    self.yzbx_test,
                                    self.features, self.multipliers)
            self.feature_size = 31
            self.max_den = max(self.feat_index.values())
            self.max_seq_len = int(24)
            print(self.max_den)


        elif dataset == 'LC':
            lc = load_LendingClub.LendingClub(file_path)
            self.df = lc.load_data()
            self.df = pd.get_dummies(self.df)
            self.df = self.df.sample(frac=0.05)
            self.features = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
                             'installment', 'annual_inc', 'pymnt_plan', 'dti', 'delinq_2yrs',
                             'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
                             'total_acc', 'initial_list_status', 'out_prncp', 'out_prncp_inv',
                             'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
                             'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                             'last_pymnt_amnt', 'collections_12_mths_ex_med', 'application_type',
                             'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
                             'term_ 36 months', 'term_ 60 months', 'grade_A',
                             'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G',
                             'sub_grade_A1', 'sub_grade_A2', 'sub_grade_A3', 'sub_grade_A4',
                             'sub_grade_A5', 'sub_grade_B1', 'sub_grade_B2', 'sub_grade_B3',
                             'sub_grade_B4', 'sub_grade_B5', 'sub_grade_C1', 'sub_grade_C2',
                             'sub_grade_C3', 'sub_grade_C4', 'sub_grade_C5', 'sub_grade_D1',
                             'sub_grade_D2', 'sub_grade_D3', 'sub_grade_D4', 'sub_grade_D5',
                             'sub_grade_E1', 'sub_grade_E2', 'sub_grade_E3', 'sub_grade_E4',
                             'sub_grade_E5', 'sub_grade_F1', 'sub_grade_F2', 'sub_grade_F3',
                             'sub_grade_F4', 'sub_grade_F5', 'sub_grade_G1', 'sub_grade_G2',
                             'sub_grade_G3', 'sub_grade_G4', 'sub_grade_G5', 'emp_length_1 year',
                             'emp_length_10+ years', 'emp_length_2 years', 'emp_length_3 years',
                             'emp_length_4 years', 'emp_length_5 years', 'emp_length_6 years',
                             'emp_length_7 years', 'emp_length_8 years', 'emp_length_9 years',
                             'emp_length_< 1 year', 'home_ownership_ANY', 'home_ownership_MORTGAGE',
                             'home_ownership_NONE', 'home_ownership_OTHER', 'home_ownership_OWN',
                             'home_ownership_RENT', 'verification_status_Not Verified',
                             'verification_status_Source Verified', 'verification_status_Verified',
                             'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation',
                             'purpose_educational', 'purpose_home_improvement', 'purpose_house',
                             'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
                             'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
                             'purpose_vacation', 'purpose_wedding']
            self.time_col_name = "time"
            self.event_col_name = "default_ind"
            self.full_columns = self.features
            self.full_columns.append(self.time_col_name)
            self.full_columns.append(self.event_col_name)
            self.df = self.df[self.full_columns]
            df_test = self.df.sample(frac=0.2)
            df_train = self.df.drop(df_test.index)
            print("test:{}, train:{}, all: {}".format(len(df_test), len(df_train),
                                                      len(df_test) + len(df_train)))
            df_train.to_csv(train_file, index=False, header=False)
            df_test.to_csv(test_file, index=False, header=False)
            self.feat_index, self.feat_index_lines, self.multipliers = feateng.build_feat_index(self.feat_index_f,
                                                                                                self.features, self.df)
            feateng.build_yzbx_data(self.feat_index, train_file, test_file,
                                    self.yzbx_train,
                                    self.yzbx_test,
                                    self.features, self.multipliers)
            self.feature_size = 111
            self.max_den = max(self.feat_index.values())
            self.max_seq_len = int(41)
            print(self.max_den)

        else:
            raise ValueError(dataset + "is not a valid dataset, please give a valid dataset name!")

    def build_model(self):
        FEATURE_SIZE = self.feature_size
        MAX_DEN = self.max_den
        EMB_DIM = 32
        BATCH_SIZE = 128
        MAX_SEQ_LEN = self.max_seq_len
        TRAING_STEPS = 10000000
        STATE_SIZE = 128
        GRAD_CLIP = 5.0
        L2_NORM = 0.001
        ADD_TIME = True
        ALPHA = 1.2  # coefficient for cross entropy
        BETA = 0.2  # coefficient for anlp
        DATA_PATH = self.file_folder
        DATA_SET = self.dataset
        TRAIN_FILE = self.yzbx_train
        TEST_FILE = self.yzbx_test
        LR = float(0.001)

        RUNNING_MODEL = BASE_RNN(EMB_DIM=EMB_DIM,
                                 FEATURE_SIZE=FEATURE_SIZE,
                                 BATCH_SIZE=BATCH_SIZE,
                                 MAX_DEN=MAX_DEN,
                                 MAX_SEQ_LEN=MAX_SEQ_LEN,
                                 TRAING_STEPS=TRAING_STEPS,
                                 STATE_SIZE=STATE_SIZE,
                                 LR=LR,
                                 GRAD_CLIP=GRAD_CLIP,
                                 L2_NORM=L2_NORM,
                                 DATA_PATH=DATA_PATH,
                                 TRAIN_FILE=TRAIN_FILE,
                                 TEST_FILE=TEST_FILE,
                                 DATA_SET=DATA_SET,
                                 ALPHA=ALPHA,
                                 BETA=BETA,
                                 ADD_TIME_FEATURE=ADD_TIME,
                                 FIND_PARAMETER=False,
                                 ANLP_LR=LR,
                                 DNN_MODEL=False,
                                 DISCOUNT=1,
                                 ONLY_TRAIN_ANLP=False,
                                 LOG_PREFIX="drsa")

        RUNNING_MODEL.create_graph()

        RUNNING_MODEL.run_model()
