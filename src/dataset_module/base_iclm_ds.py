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
        clip_processor_name: str,
        eos_token_id: int,
        bos_token_id: int,
        query_token_id: int,
        image_field: str = 'image',
    ):
        super().__init__()
        self.ice_idx_seq_list = []
        self.x_id_list = []
        self.test_sample_input_list = []
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.query_token_id = query_token_id
        self.image_field = image_field

        self.processor = CLIPProcessor.from_pretrained(clip_processor_name)
        self.index_ds = index_ds
        for idx_seq in data_list:
            idx_list = idx_seq[:-1]
            self.ice_idx_seq_list.append(list(idx_list))
            self.x_id_list.append(idx_seq[-1])

    def __getitem__(self, index):
        ice_seq_idx = self.ice_idx_seq_list[index]
        add_sp_token_seq_idx = (
            [self.bos_token_id, self.query_token_id] + ice_seq_idx + [self.eos_token_id]
        )

        test_sample_id = self.x_id_list[index]
        img = self.index_ds[test_sample_id][self.image_field]
        test_img_input = self.processor(
            images=img,
            return_tensors='pt',
        )
        return {
            'img_input': test_img_input,
            'ice_seq_idx': torch.tensor(add_sp_token_seq_idx, dtype=torch.long),
        }

    def __len__(self):
        return len(self.x_id_list)
