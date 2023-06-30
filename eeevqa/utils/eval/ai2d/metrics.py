import re
from rouge import Rouge
from sentence_transformers import util
import random
from torchmetrics import Metric
import torch
import pdb

########################
## Accuracy
########################

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _input_format(self, preds, target):
        assert type(preds) == list
        assert type(target) == list
        return preds, target
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert len(preds) == len(target)

        match_list = [int(preds[idx] == target[idx]) for idx in range(len(preds))]
        self.correct += torch.sum(match_list)
        self.total += torch.tensor(len(preds))

    def compute(self):
        return self.correct.float() / self.total