from typing import List

import datasets
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from .base_iclm_ds import BaseICLMDataset


class IdxBaseCaptionICLMDataset(BaseICLMDataset):
    def __init__(
        self,
        data_list: List,
        index_ds: datasets.Dataset,
        clip_processor_name: str,
        eos_token_id: int,
        bos_token_id: int,
        query_token_id: int,
        image_field: str,
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

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        data_dict['ice_input'] = data_dict['ice_seq_idx']
        return data_dict

    def __len__(self):
        return len(self.x_id_list)
