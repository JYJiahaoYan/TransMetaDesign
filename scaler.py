import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataScaler:
    def __init__(self, csv_path):
        self.csv = pd.read_csv(csv_path).fillna(0)

        self.para = torch.tensor(self.csv.iloc[:, 2:23].values.astype(np.float32), dtype=torch.float32)

        del self.csv

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.para)

    def get_scaler(self):
        return self.scaler


csv_path = r"data/all_data_RGBBICPic_onehot.csv"
data_scaler = DataScaler(csv_path)
para_scaler = data_scaler.get_scaler()