import json
import os
from functools import partial

import hydra
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, nn, optim, utils
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from src.load_ds_utils import load_coco_ds, load_vqav2_ds
from src.utils import collate_fn, data_split


# define the LightningModule
class LeverLM(pl.LightningModule):
    def __init__(self, lever_lm, lr, weight_decay=1e-2, warm_steps=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["lever_lm"])
        self.lever_lm = lever_lm

    def training_step(self, batch, batch_idx):
        output = self.lever_lm(**batch)
        loss = output["loss"]
        self.log(
            "train_loss", loss, batch_size=len(batch["ice_seq_idx"]), sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.lever_lm(**batch)
        loss = output["loss"]
        self.log("val_loss", loss, batch_size=len(batch["ice_seq_idx"]), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.lever_lm.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        step_batches = self.trainer.estimated_stepping_batches
        if isinstance(self.hparams.warm_steps, float):
            warm_steps = self.hparams.warm_steps * step_batches
        elif isinstance(self.hparams.warm_steps, int):
            warm_steps = self.hparams.warm_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(self.hparams.warm_steps)}"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class ICDSeqDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
    ):
        """
        dataset_para: The dataset parameters
        dataset: The *.py file name of the dataset class
        dataset_name: The dataset Class name
        """
        super().__init__()
        data_files_path = os.path.join(cfg.result_dir, "generated_data", cfg.data_files)
        with open(data_files_path, "r") as f:
            data = json.load(f)
        self.train_data_list, self.val_data_list = data_split(data, cfg.train_ratio)
        self.ds_factory = hydra.utils.instantiate(cfg.train.lever_lm_ds, _partial_=True)
        if cfg.task.task_name == "caption":
            self.index_ds = load_coco_ds(cfg, split="train")
        elif cfg.task.task_name == "vqa":
            self.index_ds = load_vqav2_ds(cfg, split="train")
        self.processor = CLIPProcessor.from_pretrained(
            cfg.train.lever_lm_model.clip_name
        )

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.trainset = self.ds_factory(
                data_list=self.train_data_list, index_ds=self.index_ds
            )
            self.valset = self.ds_factory(
                data_list=self.val_data_list, index_ds=self.index_ds
            )

    def train_dataloader(self):
        global collate_fn
        return DataLoader(
            self.trainset,
            batch_size=self.hparams.cfg.batch_size,
            num_workers=self.hparams.cfg.num_workers,
            shuffle=True,
            collate_fn=partial(collate_fn, processor=self.processor),
            pin_memory=True,
        )

    def val_dataloader(self):
        global collate_fn
        return DataLoader(
            self.valset,
            batch_size=self.hparams.cfg.batch_size,
            num_workers=self.hparams.cfg.num_workers,
            collate_fn=partial(collate_fn, processor=self.processor),
            shuffle=False,
        )


@hydra.main(version_base=None, config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    logger = WandbLogger(**cfg.wandb_args)
    tl_model_cpk_callback = ModelCheckpoint(
        filename="min_tl-{epoch}-{train_loss:.5f}-{val_loss:.5f}",
        monitor="train_loss",
        save_last=True,
        save_top_k=1,
        mode="min",
        dirpath=cfg.dirpath,
    )
    vl_model_cpk_callback = ModelCheckpoint(
        filename="min_vl-{epoch}-{train_loss:.5f}-{val_loss:.5f}",
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        mode="min",
        dirpath=cfg.dirpath,
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            LearningRateMonitor(),
            RichModelSummary(max_depth=2),
            RichProgressBar(),
            tl_model_cpk_callback,
            vl_model_cpk_callback,
        ],
        **cfg.trainer_args,
    )
    lever_lm_model = hydra.utils.instantiate(cfg.train.lever_lm_model)
    model = LeverLM(lever_lm_model, cfg.lr, cfg.weight_decay, cfg.warm_steps)
    data_module = ICDSeqDataModule(cfg)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    load_dotenv()
    main()
