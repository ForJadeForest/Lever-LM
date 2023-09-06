from itertools import chain
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


class SenImgEncodeCaptionICLMDataset(Dataset):
    def __init__(
        self,
        data_list: List,
        coco_dataset: Dataset,
        processor_name,
        eos_token_id,
        bos_token_id,
        query_token_id,
    ):
        super().__init__()
        self.ice_idx_seq_list = []
        self.x_id_list = []
        self.test_sample_text_input_list = []
        self.coco_train_dataset = coco_dataset
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.query_token_id = query_token_id

        self.processor = AutoProcessor.from_pretrained(processor_name)
        for idx_seq in data_list:
            idx_list = idx_seq[:-1]
            self.ice_idx_seq_list.append(list(idx_list))
            self.x_id_list.append(idx_seq[-1])

    def __getitem__(self, index):
        ice_seq_idx = self.ice_idx_seq_list[index]
        ice_text_list = [
            self.coco_train_dataset[i]['single_caption'] for i in ice_seq_idx
        ]

        ice_img_input = [self.coco_train_dataset[i]['image'] for i in ice_seq_idx]
        ice_img_input = [Image.open(i).convert('RGB') for i in ice_img_input]
        ice_input = self.processor(
            images=ice_img_input, text=ice_text_list, return_tensors="pt", padding=True
        )

        test_sample_id = self.x_id_list[index]
        img = self.coco_train_dataset[test_sample_id]['image']
        img = Image.open(img).convert('RGB')
        test_img_input = self.processor.image_processor(
            images=img,
            return_tensors='pt',
        )
        add_sp_token_seq_idx = (
            [self.bos_token_id, self.query_token_id] + ice_seq_idx + [self.eos_token_id]
        )
        return {
            'ice_input': ice_input,
            'img_input': test_img_input['pixel_values'],
            'ice_seq_idx': torch.tensor(add_sp_token_seq_idx, dtype=torch.long),
        }

    def __len__(self):
        return len(self.x_id_list)
