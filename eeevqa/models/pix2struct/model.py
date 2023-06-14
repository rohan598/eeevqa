from pytorch_lightning import LightningModule

from transformers import Pix2StructForConditionalGeneration, AutoProcessor

from collections.abc import Sequence, Callable

import torch

from functools import reduce

from eeevqa.utils.eval.metrics import create_result_dict, calculate_acc, calculate_rouge, get_answer_pair, get_explanation_pair, Accuracy, RougeScore

from eeevqa.utils.optimizers import WarmupCosineSchedule

class Pix2StructVanilla(LightningModule):
    def __init__(
            self,
            model_name_or_path: str = "",
            problem_list: Sequence = [],
            options:Sequence = [],
            task_name: str = "mmvqa",
            processor:Callable = None,
            learning_rate: float = 1e-5,
            adam_epsilon: float = 1e-8,
            max_new_tokens: int = 512,
            train_batch_size: int = 2,
            eval_batch_size: int = 2,
            output_format:str = "AE",
            warmup_steps:int = 318,
            total_steps:int = 636,
            cycles:float = 0.5,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.problem_list = problem_list
        self.options = options
        self.output_format = output_format

        self.processor = processor
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_name_or_path)
        self.max_new_tokens = max_new_tokens
        self.train_accuracy_metric = Accuracy()
        self.train_rouge_metric = RougeScore()
        self.val_accuracy_metric = Accuracy()
        self.val_rouge_metric = RougeScore()

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        self.save_hyperparameters()

    def forward(self, **inputs):
        input_dict = inputs
        del input_dict["sample_num"]
        return self.model(**input_dict)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]

        qids = batch.pop('sample_num')
        flattened_patches = batch.pop("flattened_patches")
        attention_mask = batch.pop("attention_mask")

        generated_ids = self.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_new_tokens=self.max_new_tokens)  
        text_predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size = self.train_batch_size)

        answer_predicted, answer_target = get_answer_pair(text_predictions, qids, self.problem_list, self.options)

        self.train_accuracy_metric.update(answer_predicted, answer_target)
        
        self.log("train_acc", self.train_accuracy_metric, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size = self.train_batch_size)

        if self.output_format != "A":
            explanation_predicted, explanation_target = get_explanation_pair(text_predictions, qids, self.problem_list)

            self.train_rouge_metric.update(explanation_predicted, explanation_target)

            self.log("train_rouge", self.train_rouge_metric, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size = self.train_batch_size)

        return loss
    
    def on_train_epoch_end(self):
        pass


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss = outputs[0]
        
        qids = batch.pop('sample_num')
        flattened_patches = batch.pop("flattened_patches")
        attention_mask = batch.pop("attention_mask")

        generated_ids = self.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_new_tokens=self.max_new_tokens)  
        text_predictions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, sync_dist=True, batch_size = self.eval_batch_size)

        answer_predicted, answer_target = get_answer_pair(text_predictions, qids, self.problem_list, self.options)

        self.val_accuracy_metric.update(answer_predicted, answer_target)
        
        self.log("val_acc", self.val_accuracy_metric, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size = self.eval_batch_size)

        if self.output_format != "A":
            explanation_predicted, explanation_target = get_explanation_pair(text_predictions, qids, self.problem_list)

            self.val_rouge_metric.update(explanation_predicted, explanation_target)
            
            self.log("val_rouge", self.val_rouge_metric, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size = self.eval_batch_size)
    
    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.model
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer, 
            warmup_steps=self.warmup_steps, 
            t_total=self.total_steps, 
            cycles = self.cycles
            )
        return {"optimizer":optimizer, "lr_scheduler":scheduler}
        # return [optimizer], []
    