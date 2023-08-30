from itertools import chain
from typing import List

import torch
from torch.utils.data import Dataset


class SentenceEncodeCaptionICLMDataset(Dataset):
    def __init__(self, data_list: List, coco_dataset, tokenizer):
        super().__init__()
        self.ice_idx_seq_list = []
        self.x_id_list = []
        self.test_sample_text_input_list = []
        self.coco_train_dataset = coco_dataset
        self.tokenizer = tokenizer
        for idx_seq in data_list:
            idx_list = idx_seq[:-1]
            self.ice_idx_seq_list.append(list(idx_list))
            self.x_id_list.append(idx_seq[-1])

    def __getitem__(self, index):
        ice_seq_idx = self.ice_idx_seq_list[index]
        ice_text_list = [
            self.coco_train_dataset[i]['single_caption'] for i in ice_seq_idx
        ]
        ice_text_input = self.tokenizer(
            ice_text_list, padding=True, return_tensors='pt'
        )
        test_sample_id = self.x_id_list[index]
        test_text_input = self.tokenizer(
            self.coco_train_dataset[test_sample_id]['single_caption'],
            return_tensors='pt',
        )

        return {
            'ice_input': ice_text_input,
            'x_input': test_text_input,
            'ice_seq_idx': torch.tensor(ice_seq_idx, dtype=torch.int32),
        }

    def __len__(self):
        return len(self.x_id_list)


if __name__ == '__main__':
    import json

    import torch
    from coco_ds import CocoDataset
    from torch.nn.utils.rnn import pad_sequence
    from transformers import BertTokenizer

    def collate_fn(batch):
        bs = len(batch)
        ice_num = batch[0]['ice_input'].input_ids.size(0)
        ice_input_ids = [item['ice_input']['input_ids'] for item in batch]
        ice_attn_masks = [item['ice_input']['attention_mask'] for item in batch]

        x_input_ids = [item['x_input']['input_ids'] for item in batch]
        x_attention_mask = [item['x_input']['attention_mask'] for item in batch]

        # padding ice text
        ice_max_len = max([i.size(1) for i in ice_input_ids])
        padded_ice_input_ids = torch.zeros(
            (bs, ice_num, ice_max_len), dtype=ice_input_ids[0].dtype
        )
        padded_ice_attn_masks = torch.zeros(
            (bs, ice_num, ice_max_len), dtype=ice_input_ids[0].dtype
        )
        for i in range(bs):
            seq_len = ice_input_ids[i].size(1)
            padded_ice_input_ids[i, :, :seq_len] = ice_input_ids[i]
            padded_ice_attn_masks[i, :, :seq_len] = ice_attn_masks[i]

        # pad x_input
        x_max_len = max(i.size(1) for i in x_input_ids)
        padded_x_input_ids = torch.zeros((bs, x_max_len), dtype=ice_input_ids[0].dtype)
        padded_x_attn_masks = torch.zeros((bs, x_max_len), dtype=ice_input_ids[0].dtype)
        for i in range(bs):
            seq_len = x_input_ids[i].size(1)
            padded_x_input_ids[i, :seq_len] = x_input_ids[i]
            padded_x_attn_masks[i, :seq_len] = x_attention_mask[i]

        return {
            'ice_input': {
                'input_ids': padded_ice_input_ids,
                'attention_mask': padded_ice_attn_masks,
            },
            'x_input': {
                'input_ids': padded_x_input_ids,
                'attention_mask': padded_x_attn_masks,
            },
            'ice_seq_idx': [item['ice_seq_idx'] for item in batch],
        }

    with open(
        'result/generated_data/open_flamingo-coco-caption-beam_size:8-few_shot:4-query_top_k:50_by_train_ds.json',
        'r',
    ) as f:
        data = json.load(f)
    data_list = [s_d for k, v in data.items() for s_d in v['id_list']]

    coco_train_dataset = CocoDataset(
        '/data/share/pyz/data/mscoco/mscoco2017/train2017',
        '/data/share/pyz/data/mscoco/mscoco2017/annotations/captions_train2017.json',
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ds = SentenceEncodeCaptionICLMDataset(data_list, coco_train_dataset, tokenizer)

    batch = [ds[i] for i in range(10)]
    data = collate_fn(batch)
    print(data)
