import os.path
from typing import Dict, List

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoProcessor


class IdxBaseCaptionICLMDataset(Dataset):
    def __init__(
        self,
        data_list: List,
        coco_dataset,
        processor_name,
        eos_token_id,
        bos_token_id,
        query_token_id,
    ):
        super().__init__()
        self.ice_idx_seq_list = []
        self.x_id_list = []
        self.test_sample_input_list = []
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.query_token_id = query_token_id

        self.img_processor = AutoProcessor.from_pretrained(processor_name)
        self.coco_train_dataset = coco_dataset
        for idx_seq in data_list:
            idx_list = idx_seq[:-1]
            self.ice_idx_seq_list.append(list(idx_list))
            self.x_id_list.append(idx_seq[-1])

    def __getitem__(self, index):
        ice_seq_idx = self.ice_idx_seq_list[index]
        ice_seq_idx = (
            [self.bos_token_id, self.query_token_id] + ice_seq_idx + [self.eos_token_id]
        )

        test_sample_id = self.x_id_list[index]
        img = self.coco_train_dataset[test_sample_id]['image']
        img = Image.open(img).convert('RGB')
        test_img_input = self.img_processor(
            images=img,
            return_tensors='pt',
        )
        return {
            'ice_input': torch.tensor(ice_seq_idx, dtype=torch.long),
            'img_input': test_img_input['pixel_values'],
            'ice_seq_idx': torch.tensor(ice_seq_idx, dtype=torch.long),
        }

    def __len__(self):
        return len(self.x_id_list)
