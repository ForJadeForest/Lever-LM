import os.path
from typing import Dict, List

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoProcessor


class IdxBaseCaptionICLMDataset(Dataset):
    def __init__(self, data_list: List, coco_dataset, img_processor_name):
        super().__init__()
        self.ice_idx_seq_list = []
        self.x_id_list = []
        self.test_sample_input_list = []
        self.img_processor = AutoProcessor.from_pretrained(img_processor_name)
        self.coco_train_dataset = coco_dataset
        for idx_seq in data_list:
            idx_list = idx_seq[:-1]
            self.ice_idx_seq_list.append(list(idx_list))
            self.x_id_list.append(idx_seq[-1])

    def __getitem__(self, index):
        ice_seq_idx = self.ice_idx_seq_list[index]

        test_sample_id = self.x_id_list[index]
        img = self.coco_train_dataset[test_sample_id]['image']
        img = Image.open(img).convert('RGB')
        test_img_input = self.img_processor(
            images=img,
            return_tensors='pt',
        )
        return {
            'ice_seq_idx': torch.tensor(ice_seq_idx, dtype=torch.long),
            'img_input': test_img_input['pixel_values'],
        }

    def __len__(self):
        return len(self.x_id_list)
