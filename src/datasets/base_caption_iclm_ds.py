import os.path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import Dataset


class BaseCaptionICLMDataset(Dataset):
    def __init__(self, data_list: List, dataset_embeddings_dict: Dict):
        super().__init__()
        self.ice_idx_seq_list = []
        self.x_id_list = []
        self.test_sample_input_list = []
        for idx_seq in data_list:
            idx_list = idx_seq[:-1]
            self.ice_idx_seq_list.append(list(idx_list))
            self.x_id_list.append(idx_seq[-1])
        self.dataset_embeddings_dict = dataset_embeddings_dict

    def __getitem__(self, index):
        data = self.ice_idx_seq_list[index]
        val_id = self.x_id_list[index]
        test_sample_embedding = self.dataset_embeddings_dict[val_id]
        return {
            'seq': torch.tensor(data, dtype=torch.int64),
            'test_sample_embedding': test_sample_embedding,
        }

    def __len__(self):
        return len(self.x_id_list)
