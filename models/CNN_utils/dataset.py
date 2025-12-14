from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class BatteryDataset(Dataset):
    """PyTorch 数据集封装 - 接收预处理后的numpy数组"""
    def __init__(self, X, y_array):
        """
        Args:
            X: numpy array of feature data
            y_array: numpy array of target values
        """

        self.X = X
            
        # 扩展维度以匹配 (N, sequence_length, 1)
        if self.X.ndim == 2:
            self.X = self.X[:, :, np.newaxis]
            
        self.y = torch.tensor(y_array, dtype=torch.float32)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]