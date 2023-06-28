from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from collections.abc import Callable

import numpy as np
import torch
import os
from transformers import AutoProcessor

from eeevqa.utils.dataloaders.raw_data import load_data_split

class VQADataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        
        self.dataset = np.load(dataset_path, mmap_mode='r', allow_pickle=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        return self.dataset[idx]

class ScienceQADataModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size:int = 1,
            eval_batch_size:int = 1,
            pickle_files_path:str = "",
            train_split:str = "minitrain",
            val_split:str = "minival",
            test_split:str = "minitest",
            num_workers:int = 8,
            **kwargs,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.pickle_files_path = pickle_files_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        train_dataset_path = os.path.join(self.pickle_files_path, f"{self.train_split}.pkl")
        val_dataset_path = os.path.join(self.pickle_files_path, f"{self.val_split}.pkl")
        test_dataset_path = os.path.join(self.pickle_files_path, f"{self.test_split}.pkl")

        self.train_dataset = VQADataset(train_dataset_path)

        self.val_dataset = VQADataset(val_dataset_path)

        self.test_dataset = VQADataset(test_dataset_path)
    
    # TODO can be made faster with torch stacking
    # TODO can be made faster by processing everything in advance
    def collator(self, batch):
        new_batch = {"flattened_patches":[],
                        "attention_mask":[], 
                        "labels":[],
                        "sample_num":[]}

        for item in batch:
            new_batch["flattened_patches"].append(item.flattened_patches)
            new_batch["attention_mask"].append(item.attention_mask)
            new_batch["labels"].append(item.output)
            new_batch["sample_num"].append(item.sample_num)

        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
        new_batch["labels"] = torch.cat(new_batch["labels"], dim=0)
        return new_batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size, collate_fn=self.collator, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, collate_fn=self.collator, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, collate_fn=self.collator, num_workers=self.num_workers)
    

## For generation & visualization purposes

def collator(batch):
    new_batch = {"flattened_patches":[],
                     "attention_mask":[], 
                     "labels":[],
                     "sample_num":[]}

    for item in batch:
        new_batch["flattened_patches"].append(item.flattened_patches)
        new_batch["attention_mask"].append(item.attention_mask)
        new_batch["labels"].append(item.output)
        new_batch["sample_num"].append(item.sample_num)

    new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
    new_batch["labels"] = torch.stack(new_batch["labels"])

    return new_batch

def create_eval_dataloader(pickle_files_path, eval_split, processor, max_patches, output_format, batch_size):
    
    dataset = VQADataset(load_data_split(pickle_files_path, eval_split),processor, max_patches, output_format)

    dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn=collator)
    return dataloader
