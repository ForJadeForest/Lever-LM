import json
import os
import random
from functools import partial

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from icd_lm.utils import data_split
from open_mmicl.lvlm_interface import FlamingoInterface
from utils import load_ds

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def calculate_kl_divergence(stu_logits, tea_logits, eps=1e-10):
    return (
        (
            tea_logits.softmax(dim=1)
            * (
                (tea_logits.softmax(dim=1) + eps).log()
                - (stu_logits.softmax(dim=1) + 1e-10).log()
            )
        )
        .sum(dim=1)
        .mean()
    )


# define the LightningModule
class FlamingoICLAdapter(pl.LightningModule):
    def __init__(self, interface, lr, weight_decay=1e-2, warm_steps=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=['interface'])
        self.interface = interface
        

        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=['Wqkv'])
        self.interface.model.lang_encoder = get_peft_model(self.interface.model.lang_encoder, peft_config)
        self.interface.model.lang_encoder.print_trainable_parameters()
        for p in self.interface.model.vision_encoder.parameters():
            p.requires_grad = False
        self.loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=self.interface.pad_token_id
        )

    def forward(self, model_input, ice_token_length):
        bs, aug_num, seq_len = model_input['lang_x'].shape
        length = (model_input['lang_x'] != self.interface.pad_token_id).sum(-1)
        query_length = length - ice_token_length
        for key in model_input:
            model_input[key] = model_input[key].reshape(-1, *model_input[key].shape[2:])

        outputs = self.interface.model(**model_input)
        # 计算ICE Seq的质量分数（Prob）
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = model_input['lang_x'][..., 1:].contiguous()
        ce_loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        ce_loss = ce_loss.view(shift_labels.size())
        
        mask_length = ice_token_length.reshape(-1)
        loss_mask = torch.zeros_like(shift_labels)  # [batch, seqlen]
        for i in range(len(loss_mask)):
            for j in range(mask_length[i] - 1, len(loss_mask[i])):
                loss_mask[i][j] = 1
        query_ce_loss = ce_loss * loss_mask
        lens = (model_input['lang_x'] != self.interface.pad_token_id).sum(-1)
        lens = lens - mask_length
        query_ce_loss = query_ce_loss.sum(-1) / lens
        scores = (-query_ce_loss).exp()
        scores = scores.reshape(bs, aug_num)
        logits = outputs.logits.reshape(bs, aug_num, seq_len, -1)
        
        loss_dict = {
                'loss' : 0.
            }
        
        
        for i in range(bs):
            best_ice_seq_idx = torch.argmax(scores[i])
            query_end_idx = ice_token_length[i] - 1 + query_length[i]
            query_label_idx = model_input['lang_x'][
                i, ice_token_length[i][0] : query_end_idx[0] + 1
            ].contiguous()
            query_label_idx = query_label_idx.unsqueeze(0).repeat(aug_num - 1, 1)

            # best_logits: 1, query_length, vocab_size
            best_logits = logits[
                i,
                best_ice_seq_idx,
                ice_token_length[i][best_ice_seq_idx]
                - 1 : query_end_idx[best_ice_seq_idx],
            ].unsqueeze(0)
            neg_logits = [
                logits[i, j, ice_token_length[i][j] - 1 : query_end_idx[j]]
                for j in range(aug_num)
                if j != best_ice_seq_idx
            ]
            # neg_logits: aug_num - 1, query_length, vocab_size
            neg_logits = torch.stack(neg_logits, dim=0)

            best_logits = best_logits.repeat(aug_num - 1, 1, 1)
            neg_num, cur_query_length, vocab_size = best_logits.shape


            soft_loss = calculate_kl_divergence(
                neg_logits.reshape(neg_num * cur_query_length, -1),
                best_logits.reshape(neg_num * cur_query_length, -1),
            )
            
            loss_dict['loss'] += soft_loss

        for key in loss_dict:
            loss_dict[key] = loss_dict[key] / bs
        # loss_dict['loss'] *= 0.1
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(**batch)
        log_dict = {'train_' + k: v for k,v in loss_dict.items()}
        self.log_dict(log_dict, batch_size=len(batch['model_input']['lang_x']), sync_dist=True, )
        
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        loss_dict = self.forward(**batch)
        log_dict = {'val_' + k: v for k,v in loss_dict.items()}
        self.log_dict(log_dict, batch_size=len(batch['model_input']['lang_x']), sync_dist=True, )
        return loss_dict['loss']

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.interface.model.parameters(),
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
                f'the warm_steps should be int or float, but got {type(self.hparams.warm_steps)}'
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class ICLAdapterDataset(Dataset):
    def __init__(self, ice_seq_list, index_ds) -> None:
        super().__init__()
        self.input_seq_list = ice_seq_list
        self.index_ds = index_ds

    def __getitem__(self, index):
        input_seq = self.input_seq_list[index]
        ice_seq = input_seq[:-1]
        aug_input_seq = [
            seq + [input_seq[-1]] for seq in self.generate_sub_permutations(ice_seq)
        ]
        data_sample = [self.index_ds[i] for i in input_seq]
        aug_data_sample_list = [
            [self.index_ds[i] for i in seq] for seq in aug_input_seq
        ]
        total_data = [data_sample] + aug_data_sample_list
        total_seq_idx = [input_seq] + aug_input_seq
        return {'data': total_data, 'seq_idx': total_seq_idx}

    def __len__(self):
        return len(self.input_seq_list)

    def generate_sub_permutations(self, x):
        """
        生成N个列表x的子排列。

        参数:
        N (int): 要生成的子排列的数量。
        x (list): 原始列表。
        k (int): 子排列中的元素数量。

        返回:
        list: 包含N个子排列的列表。
        """

        sub_permutations = []
        for _ in range(8):
            k = random.randint(2, len(x))
            sub_perm = random.sample(x, k)
            random.shuffle(sub_perm)
            sub_permutations.append(sub_perm)

        return sub_permutations


def collate_fn(data, interface):
    sample = data[0]
    aug_num = len(sample['data'])
    total_data = [i['data'] for i in data]
    bs = len(data)

    prompts = []
    ice_prompts = []
    for data in total_data:
        prompts.extend(interface.transfer_prompts(data, is_last_for_generation=False))
        ice_prompts.extend(
            [
                interface.concat_prompt(
                    t[:-1],
                    add_eos_token=False,
                    add_image_token=True,
                    is_last_for_generation=False,
                )
                for t in data
            ]
        )

    model_input = interface.prepare_input(
        prompts, add_eos_token=True, is_last_for_generation=False
    )
    for k in model_input:
        model_input[k] = model_input[k].reshape(bs, aug_num, *model_input[k].shape[1:])
    mask_length_list = [
        interface.get_input_token_num(mask_context + "<image>Output:")
        for mask_context in ice_prompts
    ]
    mask_length = torch.tensor(mask_length_list).reshape(bs, aug_num)
    return {'model_input': model_input, 'ice_token_length': mask_length}


class ICDSeqDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        interface,
    ):
        """
        dataset_para: The dataset parameters
        dataset: The *.py file name of the dataset class
        dataset_name: The dataset Class name
        """
        super().__init__()
        self.save_hyperparameters(ignore=['interface', 'ice_seq_list'])
        self.index_ds = load_ds(cfg, 'train')
        self.anchor_num = cfg.anchor_num
        ice_seq_list = self.generate_ice_seq_list()

        self.ice_seq_list = ice_seq_list
        self.train_data_list, self.val_data_list = train_test_split(
            self.ice_seq_list, test_size=cfg.test_size, random_state=cfg.seed
        )
        self.interface = interface

    def generate_ice_seq_list(
        self,
    ):
        ice_seq_list = []
        for i in range(self.anchor_num):
            k = random.randint(4, 10)
            # k = 4
            random_ice_seq = random.sample(list(range(len(self.index_ds))), k=k)
            random.shuffle(random_ice_seq)
            ice_seq_list.append(random_ice_seq)
        return ice_seq_list

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.trainset = ICLAdapterDataset(
                ice_seq_list=self.train_data_list,
                index_ds=self.index_ds,
            )
            self.valset = ICLAdapterDataset(
                ice_seq_list=self.val_data_list,
                index_ds=self.index_ds,
            )

    def train_dataloader(self):
        global collate_fn
        return DataLoader(
            self.trainset,
            batch_size=self.hparams.cfg.batch_size,
            num_workers=self.hparams.cfg.num_workers,
            shuffle=True,
            collate_fn=partial(collate_fn, interface=self.interface),
            pin_memory=True,
        )

    def val_dataloader(self):
        global collate_fn
        return DataLoader(
            self.valset,
            batch_size=self.hparams.cfg.batch_size,
            num_workers=self.hparams.cfg.num_workers,
            collate_fn=partial(collate_fn, interface=self.interface),
            shuffle=False,
        )


@hydra.main(
    version_base=None, config_path="./configs", config_name="train_icl_adapter.yaml"
)
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    logger = WandbLogger(**cfg.wandb_args)
    tl_model_cpk_callback = ModelCheckpoint(
        filename='min_tl-{epoch}-{train_loss:.5f}-{val_loss:.5f}',
        monitor='train_loss',
        save_last=False,
        save_top_k=1,
        mode='min',
        dirpath=cfg.dirpath,
    )
    vl_model_cpk_callback = ModelCheckpoint(
        filename='min_vl-{epoch}-{train_loss:.5f}-{val_loss:.5f}',
        monitor='val_loss',
        save_last=True,
        save_top_k=1,
        mode='min',
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
    interface = FlamingoInterface(
        lang_encoder_path=cfg.lvlm.lang_encoder_path,
        tokenizer_path=cfg.lvlm.tokenizer_path,
        flamingo_checkpoint_dir=cfg.lvlm.flamingo_checkpoint_dir,
        cross_attn_every_n_layers=cfg.lvlm.cross_attn_every_n_layers,
        hf_root=cfg.lvlm.hf_root,
        precision='fp32',
        device='cuda',
        prompt_template=cfg.task.template,
        column_token_map=cfg.task.column_token_map,
        icd_join_char=cfg.lvlm.icd_join_char,
        load_from_local=cfg.lvlm.load_from_local,
        instruction=cfg.task.instruction,
        init_device=None,
        image_field=cfg.task.image_field,
        label_field=cfg.task.output_column,
    )
    interface.tokenizer.padding_side = 'right'
    model = FlamingoICLAdapter(interface, cfg.lr, cfg.weight_decay, cfg.warm_steps)
    data_module = ICDSeqDataModule(cfg, interface)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    load_dotenv()
    main()
