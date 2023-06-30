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

class Ai2dQADataModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size:int = 1,
            eval_batch_size:int = 1,
            pickle_files_path:str = "",
            train_split:str = "tiny_train",
            val_split:str = "tiny_test",
            num_workers:int = 8,
            pin_memory:bool = False,
            processor:Callable = None,
            **kwargs,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.pickle_files_path = pickle_files_path
        self.train_split = train_split
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.processor = processor
        
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        train_dataset_path = os.path.join(self.pickle_files_path, f"{self.train_split}.pkl")
        val_dataset_path = os.path.join(self.pickle_files_path, f"{self.val_split}.pkl")

        self.train_dataset = VQADataset(train_dataset_path)

        self.val_dataset = VQADataset(val_dataset_path)
    
    # TODO can be made faster with torch stacking
    # TODO can be made faster by processing everything in advance
    def collator(self, batch):
        new_batch = {"flattened_patches":[],
                        "attention_mask":[], 
                        "labels":[],
                        "raw_output":[]}

        header_text_list = []
        input_image_list = []

        for item in batch:
            header_text_list.append(item.header_text)
            input_image_list.append(item.image)
            new_batch["labels"].append(item.output)
            new_batch["raw_output"].append(item.raw_output)
        
        new_batch["labels"] = torch.stack(new_batch["labels"])
        
        inputs = self.processor(images=input_image_list, text=header_text_list, return_tensors="pt", max_patches=max_patches)
            
        new_batch["flattened_patches"] = inputs["flattened_patches"]
        new_batch["attention_mask"] = inputs["attention_mask"]

        return new_batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size, collate_fn=self.collator, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, collate_fn=self.collator, num_workers=self.num_workers, pin_memory = self.pin_memory)

    def test_dataloader(self):
        pass
    