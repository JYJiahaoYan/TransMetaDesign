import random
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,Sampler
import numpy as np
from tokenlizer import Tokenizer
from dataset import BaseDataset , SpectrumDataset
import pickle
from sklearn.preprocessing import MinMaxScaler




class CustomDataLoader(LightningDataModule):
    def __init__(
        self,
        data_path,
        max_num=None,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.base_dir = Path(__file__).resolve().parent

        self.train_type, self.train_wave, self.train_structure = None, None, None
        self.val_type, self.val_wave, self.val_structure = None, None, None
        self.test_type, self.test_wave, self.test_structure = None, None, None
        self.max_num = max_num
        if isinstance(data_path, str):
            sourcr_data = pickle.load(open(data_path, "rb"))
            self.load_data(sourcr_data)
        elif isinstance(data_path, list):
            for path in data_path:
                sourcr_data = pickle.load(open(path, "rb"))
                self.load_data(sourcr_data)


        self.shuffle_and_zip_data()

    def load_data(self, sourcr_data):

        if self.train_type is None:
            self.train_type = sourcr_data["train"]["type"]
        else:
            self.train_type = np.concatenate((self.train_type, sourcr_data["train"]["type"]))
        if self.train_wave is None:
            self.train_wave = sourcr_data["train"]["wave"]
        else:
            self.train_wave = np.concatenate((self.train_wave, sourcr_data["train"]["wave"]))
        if self.train_structure is None:
            self.train_structure = sourcr_data["train"]["structure"]
        else:
            self.train_structure = np.concatenate((self.train_structure, sourcr_data["train"]["structure"]))
        
        if self.val_type is None:
            self.val_type = sourcr_data["valid"]["type"]
        else:
            self.val_type = np.concatenate((self.val_type, sourcr_data["valid"]["type"]))
        if self.val_wave is None:
            self.val_wave = sourcr_data["valid"]["wave"]
        else:
            self.val_wave = np.concatenate((self.val_wave, sourcr_data["valid"]["wave"]))
        if self.val_structure is None:
            self.val_structure = sourcr_data["valid"]["structure"]
        else:
            self.val_structure = np.concatenate((self.val_structure, sourcr_data["valid"]["structure"]))

        if self.test_type is None:
            self.test_type = sourcr_data["test"]["type"]
        else:
            self.test_type = np.concatenate((self.test_type, sourcr_data["test"]["type"]))
        if self.test_wave is None:
            self.test_wave = sourcr_data["test"]["wave"]
        else:
            self.test_wave = np.concatenate((self.test_wave, sourcr_data["test"]["wave"]))
        if self.test_structure is None:
            self.test_structure = sourcr_data["test"]["structure"]
        else:
            self.test_structure = np.concatenate((self.test_structure, sourcr_data["test"]["structure"]))


    def shuffle_and_zip_data(self):
        train_data = list(zip(self.train_type, self.train_wave, self.train_structure))
        val_data = list(zip(self.val_type, self.val_wave, self.val_structure))
        test_data = list(zip(self.test_type, self.test_wave, self.test_structure))

        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        self.train_type, self.train_wave, self.train_structure = zip(*train_data)
        self.val_type, self.val_wave, self.val_structure = zip(*val_data)
        self.test_type, self.test_wave, self.test_structure = zip(*test_data)

    def setup(self, stage=None):
        self.tokenizer = Tokenizer()
        if stage in ("fit", None):
            self.train_dataset = SpectrumDataset(
                self.train_type,
                self.train_wave,
                self.train_structure,
                self.max_num
            )
            self.val_dataset = SpectrumDataset(
                self.val_type,
                self.val_wave,
                self.val_structure,
                self.max_num
            )
        if stage in ("test", None):
            self.test_dataset = SpectrumDataset(
                self.test_type,
                self.test_wave,
                self.test_structure,
                self.max_num
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        type, wave, structure = [], [], []
        
        for item in batch:
            type.append(item[0])
            wave.append(item[1])
            structure.append(item[2])

        # scaler = MinMaxScaler(feature_range=(0, 1))
        # input_data = scaler.fit_transform(np.array(input_data).T).T

        batch_size = len(type)


        structure_encoded = [self.tokenizer.encode(o) for o in structure]
        max_output_length = max(len(o) for o in structure_encoded)
        max_input_data_length = 500
        output_dtype = torch.long
        type_encoded = [self.tokenizer.encode(i) for i in type]
        
        max_type_length = max(len(i) for i in type_encoded)

        
        type_batch_indices = torch.zeros((batch_size, max_type_length), dtype=torch.long)
        wave_batch_indices = torch.tensor(wave,dtype=torch.float)
        structure_batch_indices = torch.zeros((batch_size, max_output_length), dtype=output_dtype)

        for i in range(batch_size):
            type_indices = type_encoded[i]
            structure_indices = structure_encoded[i]
            type_batch_indices[i, :len(type_indices)] = torch.tensor(type_indices, dtype=torch.long)
            structure_batch_indices[i, :len(structure_indices)] = torch.tensor(structure_indices, dtype=output_dtype)
        
        return type_batch_indices, wave_batch_indices, structure_batch_indices


if __name__ == "__main__":
    dataloader = CustomDataLoader()
    dataloader.setup()
    train_loader = dataloader.train_dataloader()
    val_loader = dataloader.val_dataloader()
    test_loader = dataloader.test_dataloader()
    for batch in train_loader:
        print(batch)
        break
    for batch in val_loader:
        print(batch)
        break
    for batch in test_loader:
        print(batch)
        break
    print("ok")
