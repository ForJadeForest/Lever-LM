from typing import List

import datasets

from .base_lever_lm_ds import BaseLeverLMDataset


class LeverLMDataset(BaseLeverLMDataset):
    def __init__(
        self,
        data_list: List,
        index_ds: datasets.Dataset,
        index_set_length: int,
        query_image_field: str,
        query_text_field: str,
        icd_image_field: str = None,
        icd_text_field: str = None,
        threshold: float = 0.0,
        reverse_seq: bool = False,
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
            threshold=threshold,
            reverse_seq=reverse_seq,
        )
        self.icd_text_field = icd_text_field
        self.icd_image_field = icd_image_field

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        icd_seq_idx = self.icd_idx_seq_list[index]
        icd_input = {}
        if self.icd_text_field:
            icd_text_list = [self.index_ds[i][self.icd_text_field] for i in icd_seq_idx]
            icd_input["text"] = icd_text_list
        if self.icd_image_field:
            icd_img_input = [
                self.index_ds[i][self.icd_image_field] for i in icd_seq_idx
            ]
            icd_input["images"] = icd_img_input

        data_dict["icd_input"] = icd_input
        return data_dict

    def __len__(self):
        return len(self.x_id_list)
