import json
import os

import hydra
import torch
from dotenv import load_dotenv
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.datasets import CocoDataset
from src.utils import collate_fn, data_split


@torch.no_grad()
def validation(model, val_dataloader):
    model.eval()
    val_bar = tqdm(val_dataloader, desc=f'val Loss: xx.xxxx')
    for val_data in tqdm(val_bar):
        output = model(
            x_input=val_data['x_input'],
            ice_input=val_data['ice_input'],
            ice_seq_idx=val_data['ice_seq_idx'],
        )
        loss = output.loss
        val_bar.set_description(f'val Loss: {round(loss.item(), 4)}')
        total_val_loss += loss.item()

    mean_val_loss = total_val_loss / len(val_bar)
    return mean_val_loss


def train(
    cfg, train_dataloader, val_dataloader, model, fabric, optimizer, scheduler, save_dir
):
    for epoch in range(cfg.epochs):
        bar = tqdm(train_dataloader, desc=f'epoch:{epoch}-Loss: xx.xxxx')
        for data in bar:
            output = model(
                x_input=data['x_input'],
                ice_input=data['ice_input'],
                ice_seq_idx=data['ice_seq_idx'],
            )
            loss = output.loss
            fabric.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            bar.set_description(f'epoch:{epoch}-Loss: {round(loss.item(), 4)}')

            fabric.log("train_loss", loss.item(), step)
            fabric.log('lr', scheduler.get_last_lr()[0], step)
            step += 1

        val_loss = validation(model, val_dataloader)
        fabric.log("val_loss", val_loss, step)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            new_save_path = os.path.join(
                save_dir, f'epoch:{epoch}-min_loss:{round(min_val_loss, 4)}.pth'
            )
            state = {
                "model": model,
            }

            fabric.save(new_save_path, state)
            if old_save_path is None:
                old_save_path = new_save_path
            else:
                if fabric.local_rank == 0:
                    os.remove(old_save_path)
                old_save_path = new_save_path
    state = {
        "model": model,
    }
    fabric.save(
        os.path.join(save_dir, f'last-val_loss:{round(val_loss, 4)}.pth'), state
    )


@hydra.main(version_base=None, config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    save_dir = os.path.join(cfg.result_dir, 'model_cpk', cfg.ex_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cache_dir = os.path.join(cfg.result_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # 1. load the data
    data_files_path = os.path.join(cfg.result_dir, 'generated_data', cfg.data_files)
    with open(data_files_path, 'r') as f:
        data = json.load(f)
    train_data_list, val_data_list = data_split(data, cfg.train_ratio)
    train_coco_dataset = CocoDataset(
        cfg.dataset.train_coco_dataset_root, cfg.dataset.train_coco_annotation_file
    )

    iclm_model = hydra.utils.instantiate(cfg.train.iclm_model)

    train_ds = hydra.utils.instantiate(
        cfg.train.ice_seq_idx_ds,
        data_list=train_data_list,
        coco_dataset=train_coco_dataset,
    )
    val_ds = hydra.utils.instantiate(
        cfg.train.ice_seq_idx_ds,
        data_list=val_data_list,
        coco_dataset=train_coco_dataset,
    )
    logger = TensorBoardLogger(
        root_dir=os.path.join(cfg.result_dir, "logs"), name=cfg.ex_name
    )
    fabric = Fabric(
        loggers=logger,
        accelerator=cfg.device,
        devices=cfg.device_num,
        precision=cfg.precision,
    )
    fabric.launch()
    fabric.seed_everything(cfg.seed)
    train_dataloader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
    )
    optimizer = AdamW(iclm_model.parameters(), cfg.lr)
    model, optimizer = fabric.setup(iclm_model, optimizer)
    dataloader, val_dataloader = fabric.setup_dataloaders(dataloader, val_dataloader)
    total_steps = cfg.epochs * len(train_ds) // cfg.batch_size // cfg.device_num
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warm_up_ratio * total_steps,
        num_training_steps=total_steps,
    )

    train(
        cfg,
        train_dataloader,
        val_dataloader,
        model,
        fabric,
        optimizer,
        scheduler,
        save_dir,
    )


if __name__ == '__main__':
    load_dotenv()
    main()
