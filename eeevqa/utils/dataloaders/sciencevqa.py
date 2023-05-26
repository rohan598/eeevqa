from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from collections.abc import Callable

from .raw_data import load_data_split
import torch

class VQADataset(Dataset):
    def __init__(self, dataset, processor, max_patches):
        self.dataset = dataset
        self.processor = processor
        self.max_patches = max_patches

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item.image, return_tensors="pt", add_special_tokens=True, max_patches=self.max_patches)
        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = item.output
        encoding["sample_num"] = item.sample_num
        return encoding
    


def create_dataloaders(train_dataset, val_dataset, test_dataset, processor, collator,max_patches, batch_size):

    train_vqadataset = VQADataset(train_dataset, processor, max_patches)
    train_dataloader = DataLoader(train_vqadataset, shuffle=True, batch_size=batch_size, collate_fn=collator)

    val_vqadataset = VQADataset(val_dataset, processor, max_patches)
    val_dataloader = DataLoader(val_vqadataset, shuffle=True, batch_size = batch_size, collate_fn=collator)

    test_vqadataset = VQADataset(test_dataset, processor, max_patches)
    test_dataloader = DataLoader(test_vqadataset, shuffle=True, batch_size=batch_size, collate_fn=collator)

    return train_dataloader, val_dataloader, test_dataloader


class ScienceQADataModule(LightningDataModule):
    def __init__(
            self,
            model_name_or_path:str,
            task_name: str = "mmvqa",
            max_seq_length:int = 512,
            max_patches:int = 1024,
            train_batch_size:int = 1,
            eval_batch_size:int = 1,
            processor:Callable = None,
            pickle_files_path:str = "",
            train_split:str = "minitrain",
            val_split:str = "minival",
            test_split:str = "minitest",
            **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.max_patches = max_patches
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.pickle_files_path = pickle_files_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.processor = processor 

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.train_dataset = VQADataset(load_data_split(self.pickle_files_path, self.train_split), self.processor, self.max_patches)

        self.val_dataset = VQADataset(load_data_split(self.pickle_files_path, self.val_split), self.processor, self.max_patches)

        self.test_dataset = VQADataset(load_data_split(self.pickle_files_path, self.test_split), self.processor, self.max_patches)

    def collator(self, batch):
        new_batch = {"flattened_patches":[], "attention_mask":[]}
        texts = [item["text"] for item in batch]

        text_inputs = self.processor(text=texts, padding="max_length", truncation=True, return_tensors="pt", add_special_tokens=True, max_length=self.max_seq_length)

        new_batch["labels"] = text_inputs.input_ids

        for item in batch:
            new_batch["flattened_patches"].append(item["flattened_patches"])
            new_batch["attention_mask"].append(item["attention_mask"])

        new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
        new_batch["sample_num"] = [item["sample_num"] for item in batch]
        
        # TODO - add code for text context and lecture

        return new_batch

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.train_batch_size, collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=True, batch_size=self.eval_batch_size, collate_fn=self.collator)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=True, batch_size=self.eval_batch_size, collate_fn=self.collator)