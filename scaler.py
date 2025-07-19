import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataScaler:
    def __init__(self, csv_path):
        # 读取 CSV 文件并处理缺失值
        self.csv = pd.read_csv(csv_path).fillna(0)
        
        # 提取参数数据并转换为张量
        self.para = torch.tensor(self.csv.iloc[:, 2:23].values.astype(np.float32), dtype=torch.float32)
        
        # 删除 CSV 数据以释放内存
        del self.csv
        
        # 初始化并应用 MinMaxScaler
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.para)

    def get_scaler(self):
        return self.scaler


csv_path = r"data/all_data_RGBBICPic_onehot.csv"
data_scaler = DataScaler(csv_path)
para_scaler = data_scaler.get_scaler()