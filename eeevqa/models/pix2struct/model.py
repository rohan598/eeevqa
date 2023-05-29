from pytorch_lightning import LightningModule

from transformers import Pix2StructForConditionalGeneration, AutoProcessor

from collections.abc import Sequence

import torch

from functools import reduce

from eeevqa.utils.eval.evaluations import create_result_dict, calculate_acc, calculate_rouge

class Pix2StructVanilla(LightningModule):
    def __init__(
            self,
            model_name_or_path: str = "",
            problem_list: Sequence = [],
            options:Sequence = [],
            task_name: str = "mmvqa",
            learning_rate: float = 1e-5,
            adam_epsilon: float = 1e-8,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.problem_list = problem_list
        self.options = options

        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_name_or_path)
        self.accuracy_metric = calculate_acc
        self.rouge_metric = calculate_rouge

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.save_hyperparameters()

    def forward(self, **inputs):
        input_dict = inputs
        del input_dict["sample_num"]
        return self.model(**input_dict)
    
    # def forward(self, **inputs):
    #     return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]


        flattened_patches = batch.pop("flattened_patches")
        attention_mask = batch.pop("attention_mask")
        predictions = self.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)  
        text_predictions = self.processor.batch_decode(predictions, skip_special_tokens=True)

        qids = batch.pop('sample_num')
        # print(type(qids))
        # print(len(qids))
        # result_dict = create_result_dict(text_predictions, qids)

        outputs = {"loss": loss, "qids":qids , "preds":text_predictions}
        self.training_step_outputs.append(outputs)
        self.log("train_per_step_loss", loss)

        return loss
    
    def on_train_epoch_end(self):
        preds = [x["preds"] for x in self.training_step_outputs]
        preds = reduce(lambda x, y : x + y, preds)
        qids = [x["qids"] for x in self.training_step_outputs] 
        qids = reduce(lambda x, y : x + y, qids)
        result_dict = create_result_dict(preds, qids)
        loss = [x["loss"] for x in self.training_step_outputs]
        loss = reduce(lambda x, y : x + y, loss)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc",self.accuracy_metric(result_dict, self.problem_list, self.options), prog_bar=True)
        self.log("train_rouge",self.rouge_metric(result_dict, self.problem_list), prog_bar=True)

        self.training_step_outputs.clear()


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss = outputs[0]
        flattened_patches = batch.pop("flattened_patches")
        attention_mask = batch.pop("attention_mask")
        predictions = self.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)  
        text_predictions = self.processor.batch_decode(predictions, skip_special_tokens=True)

        qids = batch.pop('sample_num')

        outputs = {"loss": val_loss, "qids":qids , "preds":text_predictions}
        self.validation_step_outputs.append(outputs)

        self.log("val_per_step_loss", val_loss)

        return outputs
    
    def on_validation_epoch_end(self):
        preds = [x["preds"] for x in self.validation_step_outputs]
        preds = reduce(lambda x, y : x + y, preds)
        qids = [x["qids"] for x in self.validation_step_outputs] 
        qids = reduce(lambda x, y : x + y, qids)
        result_dict = create_result_dict(preds, qids)
        loss = [x["loss"] for x in self.validation_step_outputs]
        loss = reduce(lambda x, y : x + y, loss)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",self.accuracy_metric(result_dict, self.problem_list, self.options), prog_bar=True)
        self.log("val_rouge",self.rouge_metric(result_dict, self.problem_list), prog_bar=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        model = self.model
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)


        return [optimizer], []
        