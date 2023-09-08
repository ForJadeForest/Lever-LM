from itertools import chain
from typing import List

import datasets
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

from .base_iclm_ds import BaseICLMDataset


class SenImgEncodeICLMDataset(BaseICLMDataset):
    def __init__(
        self,
        data_list: List,
        index_ds: datasets.Dataset,
        clip_processor_name: str,
        eos_token_id: int,
        bos_token_id: int,
        query_token_id: int,
        image_field: str,
        text_field: str,
    ):
        super().__init__(
            data_list,
            index_ds,
            clip_processor_name,
            eos_token_id,
            bos_token_id,
            query_token_id,
            image_field,
        )
        self.text_field = text_field

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        ice_seq_idx = self.ice_idx_seq_list[index]
        ice_text_list = [self.index_ds[i][self.text_field] for i in ice_seq_idx]

        ice_img_input = [self.index_ds[i][self.image_field] for i in ice_seq_idx]
        ice_input = self.processor(
            images=ice_img_input, text=ice_text_list, return_tensors="pt", padding=True
        )

        data_dict['ice_input'] = ice_input
        return data_dict

    def __len__(self):
        return len(self.x_id_list)


def collate_fn(batch):
    bs = len(batch)
    ice_num = batch[0]['ice_input']['input_ids'].size(0)
    
    ice_input_ids = [item['ice_input']['input_ids'] for item in batch]
    ice_attn_masks = [item['ice_input']['attention_mask'] for item in batch]

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

    return {
        'ice_input': {
            'input_ids': padded_ice_input_ids,
            'attention_mask': padded_ice_attn_masks,
            'pixel_values': torch.cat(
                [item['ice_input']['pixel_values'] for item in batch], dim=0
            ),
        },
        'img_input': torch.cat([item['img_input']['pixel_values'] for item in batch], dim=0),
        'ice_seq_idx': torch.stack([item['ice_seq_idx'] for item in batch]),
    }