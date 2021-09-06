import pandas as pd
import io
import requests
from requests import RequestException


class M1:
    def __init__(self, file_path=None):
        if file_path is None:
            self.file_path = "https://raw.githubusercontent.com/Humboldt-WI/dissertations/main/credit_risk/Deep_learning_survival/datasets/mortgage/WideFormatMortgageAfterRemovingNull.csv"
            self.is_local = False
            print("Local files is not defined, downloading")
        else:
            self.is_local = True
            self.file_path = file_path

    def load_data(self):
        if not self.is_local:
            try:
                s = requests.get(self.file_path).content
                df = pd.read_csv(io.StringIO(s.decode('utf-8')))
            except RequestException:
                print("Error downloading files")
        else:
            df = pd.read_csv(self.file_path)
        return df
