from typing import List

import datasets

from .base_iclm_ds import BaseICLMDataset


class ICETextImageICLMDataset(BaseICLMDataset):
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
