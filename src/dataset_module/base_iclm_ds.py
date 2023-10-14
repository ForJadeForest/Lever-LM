from typing import List

import datasets
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class BaseICLMDataset(Dataset):
    def __init__(
        self,
        data_list: List,
        index_ds: datasets.Dataset,
        eos_token_id: int,
        bos_token_id: int,
        query_token_id: int,
        padding_token_id: int,
        query_image_field: str = None,
        query_text_field: str = None,
    ):
        super().__init__()
        self.ice_idx_seq_list = []
        self.x_id_list = []

        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.query_token_id = query_token_id
        self.padding_token_id = padding_token_id

        self.query_image_field = query_image_field
        self.query_text_field = query_text_field

        self.index_ds = index_ds
        self.max_length = 0
        for idx_seq in data_list:
            idx_list = idx_seq[:-1]
            self.ice_idx_seq_list.append(list(idx_list))
            self.max_length = max(len(idx_list), self.max_length)
            self.x_id_list.append(idx_seq[-1])

    def __getitem__(self, index):
        ice_seq_idx = self.ice_idx_seq_list[index]
        add_sp_token_seq_idx = (
            [self.bos_token_id, self.query_token_id] + ice_seq_idx + [self.eos_token_id]
        )
        # padding to max_length
        add_sp_token_seq_idx = add_sp_token_seq_idx + [
            self.padding_token_id for _ in range(self.max_length - len(ice_seq_idx))
        ]

        test_sample_id = self.x_id_list[index]
        query_input = {}
        if self.query_image_field:
            img = self.index_ds[test_sample_id][self.query_image_field]
            query_input['images'] = img
        if self.query_text_field:
            text = self.index_ds[test_sample_id][self.query_text_field]
            query_input['text'] = text
        return {
            'query_input': query_input,
            'ice_seq_idx': torch.tensor(add_sp_token_seq_idx, dtype=torch.long),
        }

    def __len__(self):
        return len(self.x_id_list)
