import pandas as pd
import io
import requests
from requests import RequestException


class M2:
    def __init__(self, file_path=None, load_batches=True):
        self.load_batches = load_batches
        if file_path is None:
            self.file_path = "https://raw.githubusercontent.com/Humboldt-WI/dissertations/main/credit_risk/Deep_learning_survival/datasets/data%20batches/"
            self.is_local = False
            print("Local files is not defined, downloading")
        else:
            self.is_local = True
            self.file_path = file_path

    def load_data(self):
        if not self.load_batches:
            path = self.file_path+"\\ndb1.csv"
            if not self.is_local:
                try:
                    s = requests.get(self.file_path).content
                    df = pd.read_csv(io.StringIO(s.decode('utf-8')))
                except RequestException:
                    print("Error downloading files")
            else:
                df = pd.read_csv(self.file_path)
        else:
            df = self._load_data_batches()
        df = df.drop(['label', 'payoff', 'current_year'], axis=1)
        return df

    def _load_data_batches(self):
        if not self.is_local:
            try:
                file_list = []
                for i in range(1,11):
                    df_path = self.file_path+"\\ndb"+str(i)+".csv"
                    s = requests.get(df_path)
                    df_batch = pd.read_csv(io.StringIO(s.decode('utf-8')))
                    file_list.append(df_batch)
                df = pd.concat(file_list)
            except RequestException:
                print("Error downloading files")
        else:
            file_list = []
            for i in range(1, 11):
                df_path = self.file_path + "\\ndb" + str(i) + ".csv"
                df_batch = pd.read_csv(df_path)
                file_list.append(df_batch)
            df = pd.concat(file_list)
        return df