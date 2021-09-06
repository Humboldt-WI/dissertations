import pandas as pd
import numpy as np


class LendingClub:
    def __init__(self, file_path=None):
        if file_path is None:
            raise ValueError(
                "File path not defined, please go to https://www.kaggle.com/sonujha090/xyzcorp-lendingdata?select=XYZCorp_LendingData.txt and download the data manually")
        else:
            self.file_path = file_path

    def load_data(self):
        try:
            df = pd.read_table(self.file_path, parse_dates=['issue_d'], low_memory=False)
        except FileNotFoundError:
            print("File not found in given path,please input correct file path")
        print(df.columns)
        missing_data = df.isnull().mean().sort_values(ascending=False) * 100
        columns_to_drop = sorted(list(missing_data[missing_data > 10].index))
        print("Columns with less than 90% of the data included:\n" + str(columns_to_drop))
        df = df.drop(labels=columns_to_drop, axis=1)
        df = df.dropna(axis=0, how='any')
        print(df.columns)
        df['year'] = pd.DatetimeIndex(df['issue_d']).year
        df = df[df['year'] <= 2013]
        df = df.drop(['id', 'member_id', 'zip_code', 'earliest_cr_line', 'last_credit_pull_d', 'policy_code', 'emp_title', 'title', 'addr_state', 'year'], axis=1)
        df['issue_d'] = pd.to_datetime(df['issue_d'])
        df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'])
        df['time'] = ((df.last_pymnt_d - df.issue_d) / np.timedelta64(1, 'M')).astype(int)
        df = df.drop(['last_pymnt_d', 'issue_d'], axis=1)
        df['pymnt_plan'] = df['pymnt_plan'].replace(['n', 'y'], [0, 1])
        df['initial_list_status'] = df['initial_list_status'].replace(['w', 'f'], [0, 1])
        df['application_type'] = df['application_type'].replace(['INDIVIDUAL', 'JOINT'], [0, 1])
        df = df.dropna(axis=0, how='any')
        return df
