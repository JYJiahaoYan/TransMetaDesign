import numpy as np
import os
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(
        self,
        filenames,
        wavelengths
    ):
        super().__init__()
        self.filenames = filenames
        self.wavelengths = wavelengths

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        return self.filenames[idx], self.wavelengths[idx]

class ConditionDataset(Dataset):
    def __init__(self, filenames, wavelengths, types):
        super().__init__()
        self.filenames = filenames
        self.wavelengths = wavelengths
        self.types = types

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):

        return self.filenames[item], self.wavelengths[item], self.types[item]

class SpectrumDataset(Dataset):
    def __init__(self, type, wave, structure , max_num):
        super().__init__()
        if max_num is None:
            max_num = len(type)
        self.type = type[:max_num]
        self.wave = wave[:max_num]
        self.structure = structure[:max_num]

    def __len__(self):
        return len(self.type)

    def __getitem__(self, idx):
        return self.type[idx], self.wave[idx], self.structure[idx]


if __name__ == '__main__':
    data = BaseDataset("test_data","test_data/train.txt")
    print("ok")