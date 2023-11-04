import json
import logging
import os
from functools import partial

import hydra
import torch
from dotenv import load_dotenv
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from src.load_ds_utils import load_coco_ds, load_vqav2_ds
from src.utils import collate_fn, data_split

logger = logging.getLogger(__name__)


@torch.no_grad()
def validation(model: nn.Module, val_dataloader: DataLoader, fabric: Fabric):
    model.eval()
    val_bar = tqdm(
        val_dataloader,
        desc=f'val Loss: xx.xxxx',
        disable=(fabric.local_rank != 0),
        ncols=100,
    )
    total_val_loss = 0.0
    for val_data in val_bar:
        output = model(**val_data)
        loss = output['loss']
        val_bar.set_description(f'val Loss: {round(loss.item(), 4)}')
        reduced_loss = fabric.all_reduce(loss, reduce_op='mean')
        total_val_loss += reduced_loss.item()

    mean_val_loss = total_val_loss / len(val_bar)
    return mean_val_loss


def train(
    cfg: DictConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: nn.Module,
    fabric: Fabric,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    save_dir: str,
):
    step = 0
    min_val_loss = float('inf')
    old_save_path = None
    for epoch in range(cfg.epochs):
        bar = tqdm(
            train_dataloader,
            desc=f'epoch:{epoch}-Loss: xx.xxxx',
            disable=(fabric.local_rank != 0),
            ncols=100,
        )
        train_loss = 0.
        for data in bar:
            model.train()
            output = model(**data)
            loss = output['loss']
            fabric.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            bar.set_description(f'epoch:{epoch}-Loss: {round(loss.item(), 4)}')

            fabric.log("train_loss", loss.item(), step)
            fabric.log('lr', scheduler.get_last_lr()[0], step)
            step += 1
            train_loss += loss.item()
            if step % cfg.val_step == 0:
                val_loss = validation(model, val_dataloader, fabric)
                fabric.log("val_loss", val_loss, step)
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    new_save_path = os.path.join(
                        save_dir,
                        f'{epoch=}-{step=}-min_loss:{round(min_val_loss, 4)}.pth',
                    )
                    state = {"model": model}

                    fabric.save(new_save_path, state)
                    if fabric.local_rank == 0:
                        logger.info(f'new model has been saved in {new_save_path}')

                    if old_save_path is None:
                        old_save_path = new_save_path
                    else:
                        if fabric.local_rank == 0:
                            os.remove(old_save_path)
                            logger.info(f'old model has been deleted: {old_save_path}')
                        old_save_path = new_save_path
        logger.info('=' * 20 + f'{epoch} Epoch Done! ')
        train_loss /= len(bar)
        if cfg.save_nper_epoch and epoch % cfg.save_nper_epoch == 0:
            new_save_path = os.path.join(
                save_dir,
                f'{epoch=}-{step=}-{train_loss=}.pth',
            )
            state = {"model": model}
            fabric.save(new_save_path, state)


    val_loss = validation(model, val_dataloader, fabric)
    fabric.log("val_loss", val_loss, step)
    state = {"model": model}
    last_model_path = os.path.join(save_dir, f'last-val_loss:{round(val_loss, 4)}.pth')
    fabric.save(last_model_path, state)
    if fabric.local_rank == 0:
        logger.info(f'last model has been saved: {last_model_path}')


@hydra.main(version_base=None, config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    global collate_fn
    logger.info(f'{cfg=}')
    save_dir = os.path.join(cfg.result_dir, 'model_cpk', cfg.task.task_name, cfg.ex_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cache_dir = os.path.join(cfg.result_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # 1. load the ice_seq data
    data_files_path = os.path.join(cfg.result_dir, 'generated_data', cfg.data_files)
    with open(data_files_path, 'r') as f:
        data = json.load(f)
    train_data_list, val_data_list = data_split(data, cfg.train_ratio)

    # load the index dataset
    if cfg.task.task_name == 'caption':
        index_ds = load_coco_ds(cfg, split='train')
    elif cfg.task.task_name == 'vqa':
        index_ds = load_vqav2_ds(cfg, split='train')

    iclm_model = hydra.utils.instantiate(cfg.train.iclm_model)
    logger.info(f'model: {type(iclm_model)} load succese')
    ds_factory = hydra.utils.instantiate(cfg.train.iclm_ds, _partial_=True)

    train_ds = ds_factory(data_list=train_data_list, index_ds=index_ds)
    logger.info('train_ds load success')
    val_ds = ds_factory(data_list=val_data_list, index_ds=index_ds)
    logger.info('val_ds load success')

    tensorboard_logger = TensorBoardLogger(
        root_dir=os.path.join(cfg.result_dir, "tensorboard-logs"), name=cfg.ex_name
    )

    fabric = Fabric(
        loggers=tensorboard_logger,
        accelerator=cfg.device,
        devices=cfg.device_num,
        precision=cfg.precision,
    )
    fabric.launch()
    fabric.seed_everything(cfg.seed)
    processor = CLIPProcessor.from_pretrained(cfg.train.iclm_model.clip_name)
    collate_fn = partial(collate_fn, processor=processor)
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
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )
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
    load_dotenv(override=True)
    main()
