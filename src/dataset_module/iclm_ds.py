from typing import List

import datasets

from .base_lever_lm_ds import BaseICLMDataset


class ICLMDataset(BaseICLMDataset):
    def __init__(
        self,
        data_list: List,
        index_ds: datasets.Dataset,
        index_set_length: int,
        query_image_field: str,
        query_text_field: str,
        ice_image_field: str = None,
        ice_text_field: str = None,
    ):
        eos_token_id = index_set_length
        bos_token_id = index_set_length + 1
        query_token_id = index_set_length + 2
        super().__init__(
            data_list,
            index_ds,
            eos_token_id,
            bos_token_id,
            query_token_id,
            query_image_field,
            query_text_field,
        )
        self.ice_text_field = ice_text_field
        self.ice_image_field = ice_image_field

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        ice_seq_idx = self.ice_idx_seq_list[index]
        ice_input = {}
        if self.ice_text_field:
            ice_text_list = [self.index_ds[i][self.ice_text_field] for i in ice_seq_idx]
            ice_input["text"] = ice_text_list
        if self.ice_image_field:
            ice_img_input = [
                self.index_ds[i][self.ice_image_field] for i in ice_seq_idx
            ]
            ice_input["images"] = ice_img_input

        data_dict["ice_input"] = ice_input
        return data_dict

    def __len__(self):
        return len(self.x_id_list)
