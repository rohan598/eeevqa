import os
from collections import namedtuple

from sys import platform
from datetime import datetime

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch

from transformers import AutoProcessor

from utils.args import parse_args
from utils.dataloaders.sciencevqa import ScienceQADataModule
from utils.dataloaders.raw_data import read_captions, read_problem_list
from utils.eval.evaluations import create_result_dict, calculate_acc, calculate_rouge

from models.pix2struct.model import Pix2StructVanilla

if __name__ == '__main__':
    seed_everything(42)
    
    print("----- Parsed Arguments -----")
    args = parse_args()

    print("----- Read Dataset -----") 
    if args.task_name == "univqa":
        ScienceQA = namedtuple("ScienceQA", "sample_num header_text image image_mean image_std output")
    else:
        ScienceQA = namedtuple("ScienceQA", "sample_num header_text image text_context lecture image_mean image_std output")

    captions_dict = read_captions(args.data_root, args.captions_filename)
    problem_list = read_problem_list(args.json_files_path, args.problems_filename)

    print("----- Setup Lightning Data Module -----")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.base_model_name)
    if device=="cuda":
        processor.image_processor.convert_fp16 = True
     
    train_split =  args.train_split
    val_split =  args.val_split
    test_split =  args.test_split

    if args.dummy_run == "yes":
        train_split = "tiny_train" 
        val_split = "tiny_val"
        test_split = "tiny_test"

    sdm = ScienceQADataModule(
            model_name_or_path=args.base_model_name,
            max_seq_length = args.max_tokens,
            max_patches = args.max_patches,
            train_batch_size = args.train_batch_size,
            eval_batch_size = args.eval_batch_size,
            processor = processor,
            pickle_files_path = os.path.join(args.pickle_files_path, args.data_type),
            train_split =  train_split,
            val_split =  val_split,
            test_split =  test_split,
    )
    
    print("----- Fitting Lightning Data Module -----")
    sdm.setup("fit")

    print("----- Setup Lightning Model -----")
    model = Pix2StructVanilla(
            model_name_or_path = args.base_model_name,
            problem_list = problem_list,
            options = args.options,
            task_name = args.task_name,
            learning_rate = args.learning_rate,
            adam_epsilon = 1e-8,
            train_batch_size = args.train_batch_size,
            eval_batch_size = args.eval_batch_size,
    )

    print("----- Setup Model Callbacks -----")
    checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_root,
            filename='{epoch}-{val_loss:.2f}-{val_metric:.2f}',
            monitor='val_acc',
            mode = "max",
            save_top_k = 1,
            save_last = True,
            every_n_epochs = args.save_every,
    )
    wandb_logger = WandbLogger(
        project="mmvqa",
        name = f"run_unimodal_{train_split}_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}",
        save_dir = args.output_root
    )

    print("----- Setup and fit Trainer -----")
    if platform == "darwin":
        trainer = Trainer(
        max_epochs=5,
        accelerator="mps",
        devices=1,  
        callbacks=[checkpoint_callback],
        logger = wandb_logger,
        )

    else:
        trainer = Trainer(
        max_epochs=args.epoch,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  
        callbacks=[checkpoint_callback],
        logger = wandb_logger,
        )

    trainer.fit(model, datamodule=sdm)


'''
TODO:

Series Task-1:
1-> Change structure of the code to follow a package
2-> Run stats on token size for lecture and text context combined and get visual results for it
3-> Write code for combining bert and pix2struct
    -> experiment with MLP, MHA and attention


Series Task-2:
1-> Run baseline on mini dataset

Series Task-3: 
1-> Test the structure and generate unimodal train dataset (dep Task-1:1)
2-> Test the structure and generate multimodal dataset (dep Task-1:2)


'''