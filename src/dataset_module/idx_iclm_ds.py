from typing import List

import datasets

from .base_lever_lm_ds import BaseICLMDataset


class IdxICLMDataset(BaseICLMDataset):
    def __init__(
        self,
        data_list: List,
        index_ds: datasets.Dataset,
        clip_processor_name: str,
        index_set_length: int,
        image_field: str,
    ):
        eos_token_id = index_set_length
        bos_token_id = index_set_length + 1
        query_token_id = index_set_length + 2
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
        data_dict["ice_input"] = data_dict["ice_seq_idx"]
        return data_dict

    def __len__(self):
        return len(self.x_id_list)
