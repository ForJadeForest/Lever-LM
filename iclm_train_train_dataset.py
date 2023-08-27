import argparse
import gc
import json
import os
import random

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Config, get_cosine_schedule_with_warmup

from src.datasets import BaseCaptionICLMDataset, CocoDataset
from src.models import BaseCaptionICLM
from src.utils import encode_dataset


def data_collator(data_list):
    max_len = max([item['seq'].size(0) for item in data_list])

    # 初始化一个列表来保存padding后的tensors和attention masks
    padded_tensors = []
    attention_masks = []

    for item in data_list:
        seq = item['seq']
        seq_len = seq.size(0)

        # 创建一个新tensor，使用0进行padding
        padded_tensor = torch.zeros(max_len, dtype=seq.dtype)
        # 将原tensor复制到新tensor
        padded_tensor[:seq_len] = seq
        padded_tensors.append(padded_tensor)

        # 创建一个新的attention mask，将在seq长度范围内赋值为1，其他地方为0
        attention_mask = torch.zeros(max_len, dtype=torch.long)
        attention_mask[:seq_len] = 1
        attention_masks.append(attention_mask)

    # 使用torch.stack将列表中的所有tensor堆叠在新的维度上
    stacked_seq_input_ids = torch.stack(padded_tensors)
    stacked_seq_masks = torch.stack(attention_masks)

    tensor_input_dict = {
        'seq_input_ids': stacked_seq_input_ids,
        'seq_attention_mask': stacked_seq_masks,
        'test_sample_embedding': torch.stack(
            [d['test_sample_embedding'] for d in data_list]
        ),
    }
    return tensor_input_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name. Currently only `OpenFlamingo` is supported.",
        default="open_flamingo",
    )
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
    parser.add_argument('--emb_dim', type=int, default=4096)

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
    parser.add_argument(
        '--dataset_encoder_bs', type=int, help='the batch size when dataset encoding'
    )
    parser.add_argument('--pool_method', type=str, default='mean')
    # Other args
    parser.add_argument('--ex_name', type=str, help='the experiments name')
    parser.add_argument(
        '--result_dir', type=str, help='the result dir for model and cache save'
    )
    parser.add_argument(
        '--data_files', type=str, help='the file name of generated data by InfoScore'
    )
    parser.add_argument(
        '--data_split_cache_path',
        type=str,
        help='To save the test data train and val split',
    )

    return parser.parse_args()


def data_split(generated_data, args):
    # 获得有多少条test数据
    test_dataset_id_set = {
        v[-1] for d in generated_data for v in generated_data[d]['id_list']
    }
    test_dataset_len = len(test_dataset_id_set)

    # 计算多少test数据用于训练 剩下部分用于监督val loss
    train_data_len = int(args.train_ratio * test_dataset_len)
    train_idx_set = set(random.sample(test_dataset_id_set, train_data_len))
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

    # 缓存dataset embedding
    train_cache_emb_path = os.path.join(
        cache_dir, f'coco_train_flamingo_mean_feature.pth'
    )

    dataset_embedding_dict = encode_dataset(
        train_coco_dataset,
        args.pool_method,
        train_cache_emb_path,
        args.dataset_encoder_bs,
        args.lang_encoder_path,
        args.tokenizer_path,
        args.flamingo_checkpoint_path,
        args.cross_attn_every_n_layers,
        args.hf_root,
        args.precision,
        args.device,
    )

    train_dataset = BaseCaptionICLMDataset(train_data_list, dataset_embedding_dict)
    val_dataset = BaseCaptionICLMDataset(val_data_list, dataset_embedding_dict)

    model = BaseCaptionICLM(lm_config, args.emb_dim).to(args.device)

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator,
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
            output = model(
                test_sample_embedding=data['test_sample_embedding'].to(args.device),
                seq_input_ids=data['seq_input_ids'].to(args.device),
                seq_attention_mask=data['seq_attention_mask'].to(args.device),
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
                output = model(
                    test_sample_embedding=val_data['test_sample_embedding'].to(
                        args.device
                    ),
                    seq_input_ids=val_data['seq_input_ids'].to(args.device),
                    seq_attention_mask=val_data['seq_attention_mask'].to(args.device),
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
        model, os.path.join(save_dir, f'last-val_loss:{round(min_val_loss, 4)}.pth')
    )
