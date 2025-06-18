import copy

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchio
import numpy as np

class ACCDataset_DDP(Dataset):
    def __init__(self,meta):
        self.meta = meta
        self.data = {}
    def __len__(self):
        return len(self.meta)
    def __load_single_acc__(self,path):
        return np.loadtxt(path).reshape(1,-1)
    def __getitem__(self, idx):
        meta = self.meta.iloc[idx]
        if not idx in self.data.keys():
            data = self.__load_single_acc__(meta['PATH'])
            self.data[idx] = data
        return (meta['LABEL'], self.data[idx])
    def print(self,path):
        df = {'index':[i for i in self.data.keys()]}
        pd.DataFrame(df).to_csv(path, index=False, header=True)