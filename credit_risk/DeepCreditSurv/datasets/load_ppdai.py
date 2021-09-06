

import pandas as pd
import numpy as np


class ppdai:
    def __init__(self, file_path=None):
        if file_path is None:
            raise ValueError(
                "File path not defined, please go to https://www.kesci.com/mw/dataset/58c614aab84b2c48165a262d and download the data manually")
        else:
            self.file_path = file_path

    def load_data(self):
        try:
            lc_path = self.file_path + "\\LC.csv"
            lp_path = self.file_path + "\\LP.csv"
            df_ppd_lc = pd.read_csv(lc_path)
            df_ppd_lp = pd.read_csv(lp_path)
        except FileNotFoundError:
            print("File not found in given path,please input correct file path")
        df_ppd_lc.columns = ['ListingId', 'LoanValue', 'LoanPeriod',
                             'LoanRate', 'LoanDate', 'BorrowerRating',
                             'LoanType', 'IsFirstTime', 'Age', 'Gender',
                             'PhoneVerified', 'RegistrationVerified',
                             'VideoVerified', 'DegreeVerified', 'CreditVerified',
                             'TaobaoVerified', 'HistoryLoans', 'HistoryLoanValue',
                             'TotalLoanstopay', 'HistoryNormalrepaymonths',
                             'HistoryDefaultMonths']  # some columns translation work
        df_ppd_lc = df_ppd_lc.replace(['是', '否', '成功认证', '未成功认证', '男', '女'], [1, 0, 1, 0, 1, 0])
        df_ppd_lc['LoanType'] = df_ppd_lc['LoanType'].replace(['普通', 'APP闪电', '其他', '电商'],
                                                              ['Normal', 'APPexpress', 'Others', 'E-commerce'])
        df_ppd_lp.columns = ['ListingId', 'Period', 'RepaymentStatus', 'Loanstopay', 'interesttopay',
                             'RemainingPrinciple', 'RemainingInterest', 'DueDate', 'RepaymentDate', 'recorddate']
        df_ppd_lp['RepaymentStatus'] = df_ppd_lp['RepaymentStatus'].replace([0, 1, 2, 3, 4], [0, 0, 1, 0, 0])
        df_ppd_lp = df_ppd_lp.drop(['recorddate'], axis=1)
        df = df_ppd_lp.join(df_ppd_lc.set_index('ListingId'), on='ListingId')
        df = df.drop(['ListingId', 'DueDate', 'RepaymentDate', 'LoanDate'], axis=1)
        df = df[:100000]
        return df
