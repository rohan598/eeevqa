import os
from collections import namedtuple

from sys import platform
from datetime import datetime

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import DeviceStatsMonitor

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger 
import torch

from transformers import AutoProcessor

from eeevqa.utils.args import parse_args, parse_boolean
from eeevqa.utils.dataloaders.ai2d import Ai2dQADataModule
from eeevqa.utils.dataloaders.raw_data import read_json_file

from eeevqa.models.ai2d.pix2struct import Pix2StructVanilla

if __name__ == '__main__':
    # Global settings
    seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    args = parse_args()
    print("----- Parsed Arguments -----")

    TrainQA = namedtuple("TrainQA", "sample_num header_text image flattened_patches attention_mask raw_output output")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.base_model_name)
    print("----- Lightning Data Module Setup -----")

    train_split =  args.train_split
    val_split =  args.val_split
    log_every_n_steps = args.log_every_n_steps

    pickle_files_path = os.path.join(args.data_root, args.pickle_files_dir)
    
    sdm = Ai2dQADataModule(
            train_batch_size = args.train_batch_size,
            eval_batch_size = args.eval_batch_size,
            pickle_files_path = pickle_files_path,
            train_split =  train_split,
            val_split =  val_split,
            num_workers = args.num_workers,
            pin_memory = True if device == "cuda" else False,
            processor = processor
    )

    sdm.setup("fit")
    print("----- Lightning Data Module Fitted -----")

    skip_scheduler = parse_boolean(args.skip_scheduler)

    model = Pix2StructVanilla(
            model_name_or_path = args.base_model_name,
            options = args.options,
            processor=processor,
            learning_rate = args.learning_rate,
            max_new_tokens = args.max_new_tokens,
            train_batch_size = args.train_batch_size,
            eval_batch_size = args.eval_batch_size,
            skip_scheduler=skip_scheduler,
            warmup_steps = args.warmup_steps,
            total_steps = args.total_steps,
            cycles = args.cycles
    )
    print("----- Lightning Model Setup -----")

    checkpoint_dir = os.path.join(args.output_root, args.checkpoint_dir, args.train_split)
    if os.path.exists(checkpoint_dir)==False:
        os.makedirs(checkpoint_dir)

    checkpoint_callback = ModelCheckpoint(
            dirpath= checkpoint_dir,
            filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
            monitor='val_acc',
            mode = "max",
            save_top_k = 1,
            save_last = False,
            every_n_epochs = args.save_every_n_epoch,
    )
    # logger = WandbLogger(
    #     project="pix2struct",
    #     name = f"run_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}",
    #     save_dir = args.output_root
    # )

    tensorboard_path = os.path.join(args.output_root,"tensorboard")

    if os.path.exists(tensorboard_path)==False:
        os.makedirs(tensorboard_path)
    
    logger = TensorBoardLogger(
        name = f"run_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}",
        save_dir = os.path.join(tensorboard_path)
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stopping = EarlyStopping(monitor=args.es_monitor, min_delta=args.es_min_delta, patience=args.es_patience, verbose=False, mode=args.es_mode)
    print("----- Model Callbacks Setup -----")

    if platform == "darwin":
        trainer = Trainer(
        max_epochs=5,
        accelerator="mps",
        devices=1,  
        callbacks=[checkpoint_callback, lr_monitor],
        logger = wandb_logger,
        )

    else:
        trainer = Trainer(
            max_epochs=args.epoch,
            accelerator="gpu",
            devices=args.gpu_cnt if torch.cuda.is_available() else None,
            strategy="ddp",  
            precision="bf16-mixed",
            callbacks=[checkpoint_callback, lr_monitor, early_stopping],
            logger = logger,
            log_every_n_steps = log_every_n_steps,
            profiler="simple"
        )
    print("----- Trainer Setup -----")

    trainer.fit(model, datamodule=sdm)
    print("----- Trainer Fitted -----")