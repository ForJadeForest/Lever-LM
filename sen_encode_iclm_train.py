import argparse
import gc
import json
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, GPT2Config, get_cosine_schedule_with_warmup

from src.datasets import CocoDataset, SentenceEncodeCaptionICLMDataset
from src.models import SenEncodeCaptionICLM


def collate_fn(batch):
    bs = len(batch)
    ice_num = batch[0]['ice_input'].input_ids.size(0)
    ice_input_ids = [item['ice_input']['input_ids'] for item in batch]
    ice_attn_masks = [item['ice_input']['attention_mask'] for item in batch]

    x_input_ids = [item['x_input']['input_ids'] for item in batch]
    x_attention_mask = [item['x_input']['attention_mask'] for item in batch]

    # padding ice text
    ice_max_len = max([i.size(1) for i in ice_input_ids])
    padded_ice_input_ids = torch.zeros(
        (bs, ice_num, ice_max_len), dtype=ice_input_ids[0].dtype
    )
    padded_ice_attn_masks = torch.zeros(
        (bs, ice_num, ice_max_len), dtype=ice_input_ids[0].dtype
    )
    for i in range(bs):
        seq_len = ice_input_ids[i].size(1)
        padded_ice_input_ids[i, :, :seq_len] = ice_input_ids[i]
        padded_ice_attn_masks[i, :, :seq_len] = ice_attn_masks[i]

    # pad x_input
    x_max_len = max(i.size(1) for i in x_input_ids)
    padded_x_input_ids = torch.zeros((bs, x_max_len), dtype=ice_input_ids[0].dtype)
    padded_x_attn_masks = torch.zeros((bs, x_max_len), dtype=ice_input_ids[0].dtype)
    for i in range(bs):
        seq_len = x_input_ids[i].size(1)
        padded_x_input_ids[i, :seq_len] = x_input_ids[i]
        padded_x_attn_masks[i, :seq_len] = x_attention_mask[i]

    return {
        'ice_input': {
            'input_ids': padded_ice_input_ids,
            'attention_mask': padded_ice_attn_masks,
        },
        'x_input': {
            'input_ids': padded_x_input_ids,
            'attention_mask': padded_x_attn_masks,
        },
        'ice_seq_idx': torch.stack([item['ice_seq_idx'] for item in batch], dim=0),
    }


def get_args():
    parser = argparse.ArgumentParser()

    # flamingo args for dataset encode
    parser.add_argument("--lang_encoder_path", type=str, help="The lang_encoder_path")
    parser.add_argument('--tokenizer_path', type=str, help='the tokenizer_path ')
    parser.add_argument(
        '--flamingo_checkpoint_path', type=str, help='The checkpoint of open_flamingo'
    )
    parser.add_argument('--cross_attn_every_n_layers', type=int, help='the ')
    parser.add_argument('--hf_root', type=str, help='the flamingo version')

    # ICLM args
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)

    # training hyperparameters
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)
    parser.add_argument('--precision', type=str, default='bf16')

    # dataset args
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--train_coco_dataset_root', type=str)
    parser.add_argument('--train_coco_annotation_file', type=str)
    # Other args
    parser.add_argument('--ex_name', type=str, help='the experiments name')
    parser.add_argument(
        '--result_dir', type=str, help='the result dir for model and cache save'
    )
    parser.add_argument(
        '--data_files', type=str, help='the file name of generated data by InfoScore'
    )
    parser.add_argument('--seed', type=int, help='The random seed', default=42)

    return parser.parse_args()


def data_split(generated_data, args):
    # 获得有多少条test数据
    test_dataset_id_set = {
        v[-1] for d in generated_data for v in generated_data[d]['id_list']
    }
    test_dataset_len = len(test_dataset_id_set)

    # 计算多少test数据用于训练 剩下部分用于监督val loss
    train_data_len = int(args.train_ratio * test_dataset_len)
    train_idx_set = set(random.sample(list(test_dataset_id_set), train_data_len))
    val_idx_set = test_dataset_id_set - train_idx_set

    train_data_list = list()
    val_data_list = list()
    for d in data:
        for i in data[d]['id_list']:
            if int(i[-1]) in train_idx_set:
                train_data_list.append(i)
            elif int(i[-1]) in val_idx_set:
                val_data_list.append(i)
            else:
                raise ValueError()

    print(f'the train size {len(train_data_list)}, the test size {len(val_data_list)}')
    return train_data_list, val_data_list


if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)

    save_dir = os.path.join(args.result_dir, 'model_cpk', args.ex_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cache_dir = os.path.join(args.result_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    lm_config = GPT2Config(
        vocab_size=args.vocab_size,
        n_layer=args.n_layers,
        n_embd=args.n_embd,
        n_head=args.n_head,
    )
    data_files_path = os.path.join(args.result_dir, 'generated_data', args.data_files)
    with open(data_files_path, 'r') as f:
        data = json.load(f)

    train_data_list, val_data_list = data_split(data, args)
    # 加载数据集
    train_coco_dataset = CocoDataset(
        args.train_coco_dataset_root, args.train_coco_annotation_file
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = SentenceEncodeCaptionICLMDataset(
        train_data_list, train_coco_dataset, tokenizer
    )
    val_dataset = SentenceEncodeCaptionICLMDataset(
        val_data_list, train_coco_dataset, tokenizer
    )

    model = SenEncodeCaptionICLM(
        lm_config,
    ).to(args.device)

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
    )

    optimizer = AdamW(model.parameters(), args.lr)
    total_steps = args.epochs * len(train_dataset) // args.batch_size
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warm_up_ratio * total_steps,
        num_training_steps=total_steps,
    )

    min_val_loss = float('inf')
    old_save_path = None
    for epoch in range(args.epochs):
        bar = tqdm(dataloader, desc=f'epoch:{epoch}-Loss: xx.xxxx')
        for data in bar:
            x_input = {k: v.to(args.device) for k, v in data['x_input'].items()}
            ice_input = {k: v.to(args.device) for k, v in data['ice_input'].items()}

            output = model(
                x_input=x_input,
                ice_input=ice_input,
                ice_seq_idx=data['ice_seq_idx'].to(args.device),
            )
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            bar.set_description(f'epoch:{epoch}-Loss: {round(loss.item(), 4)}')

        total_val_loss = 0.0

        with torch.no_grad():
            model.eval()
            val_bar = tqdm(val_dataloader, desc=f'val Loss: xx.xxxx')
            for val_data in tqdm(val_bar):
                x_input = {k: v.to(args.device) for k, v in val_data['x_input'].items()}
                ice_input = {
                    k: v.to(args.device) for k, v in val_data['ice_input'].items()
                }

                output = model(
                    x_input=x_input,
                    ice_input=ice_input,
                    ice_seq_idx=val_data['ice_seq_idx'].to(args.device),
                )
                loss = output.loss
                bar.set_description(f'val Loss: {round(loss.item(), 4)}')
                total_val_loss += loss.item()
        mean_val_loss = total_val_loss / len(val_bar)
        val_bar.write(f'mean val loss: {total_val_loss / len(val_bar)}')

        if mean_val_loss < min_val_loss:
            min_val_loss = mean_val_loss
            new_save_path = os.path.join(
                save_dir, f'epoch:{epoch}-min_loss:{round(min_val_loss, 4)}.pth'
            )
            torch.save(model, new_save_path)
            if old_save_path is None:
                old_save_path = new_save_path
            else:
                os.remove(old_save_path)
                old_save_path = new_save_path
    torch.save(
        model, os.path.join(save_dir, f'last-val_loss:{round(mean_val_loss, 4)}.pth')
    )
