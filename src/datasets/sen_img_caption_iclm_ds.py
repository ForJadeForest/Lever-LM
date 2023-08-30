from itertools import chain
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset


class SenImgEncodeCaptionICLMDataset(Dataset):
    def __init__(self, data_list: List, coco_dataset, tokenizer, img_processor):
        super().__init__()
        self.ice_idx_seq_list = []
        self.x_id_list = []
        self.test_sample_text_input_list = []
        self.coco_train_dataset = coco_dataset
        self.tokenizer = tokenizer
        self.img_processor = img_processor
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
        img = self.coco_train_dataset[test_sample_id]['image']
        img = Image.open(img).convert('RGB')
        test_img_input = self.img_processor(
            images=img,
            return_tensors='pt',
        )

        return {
            'ice_input': ice_text_input,
            'x_input': test_img_input['pixel_values'],
            'ice_seq_idx': torch.tensor(ice_seq_idx, dtype=torch.int32),
        }

    def __len__(self):
        return len(self.x_id_list)


if __name__ == '__main__':
    import json

    import torch
    from coco_ds import CocoDataset
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoProcessor, AutoTokenizer

    def collate_fn(batch):
        bs = len(batch)
        ice_num = batch[0]['ice_input'].input_ids.size(0)
        ice_input_ids = [item['ice_input']['input_ids'] for item in batch]
        ice_attn_masks = [item['ice_input']['attention_mask'] for item in batch]

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

        return {
            'ice_input': {
                'input_ids': padded_ice_input_ids,
                'attention_mask': padded_ice_attn_masks,
            },
            'x_input': torch.cat([item['x_input'] for item in batch], dim=0),
            'ice_seq_idx': torch.stack([item['ice_seq_idx'] for item in batch]),
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
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    img_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    ds = SenImgEncodeCaptionICLMDataset(
        data_list, coco_train_dataset, tokenizer, img_processor
    )

    batch = [ds[i] for i in range(10)]
    data = collate_fn(batch)
    print(data)
