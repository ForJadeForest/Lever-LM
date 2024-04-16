import json
import os
from functools import partial

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import optim
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from icd_lm.utils import data_split, collate_fn
from utils import load_ds


# define the LightningModule
class ICDLM(pl.LightningModule):
    def __init__(self, icd_lm, lr, weight_decay=1e-2, warm_steps=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["icd_lm"])
        self.icd_lm = icd_lm

    def training_step(self, batch, batch_idx):
        output = self.icd_lm(**batch)
        loss = output["loss"]
        self.log(
            "train_loss", loss, batch_size=len(batch["icd_seq_idx"]), sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.icd_lm(**batch)
        loss = output["loss"]
        self.log("val_loss", loss, batch_size=len(batch["icd_seq_idx"]), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.icd_lm.parameters(),
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
        self.ds_factory = hydra.utils.instantiate(cfg.train.icd_lm_ds, _partial_=True)
        self.index_ds = load_ds(cfg, "train")
        self.processor = CLIPProcessor.from_pretrained(cfg.train.icd_lm.clip_name)

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
        save_last=False,
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
    icd_lm = hydra.utils.instantiate(cfg.train.icd_lm)
    model = ICDLM(icd_lm, cfg.lr, cfg.weight_decay, cfg.warm_steps)
    data_module = ICDSeqDataModule(cfg)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    load_dotenv()
    main()
