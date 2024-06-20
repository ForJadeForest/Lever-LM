import json
import os
import random
from typing import Any

from loguru import logger


class BaseSampler:
    def __init__(
        self,
        candidate_num,
        sampler_name,
        cache_dir,
        anchor_sample_num,
        index_ds_len,
        overwrite,
        dataset_name,
        other_info='',
        anchor_idx_list=None
    ) -> None:
        self.candidate_num = candidate_num
        self.sampler_name = sampler_name
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.overwrite = overwrite
        self.anchor_sample_num = anchor_sample_num
        self.anchor_set_cache_fn = os.path.join(
            cache_dir, f'{dataset_name}-anchor_sample_num:{self.anchor_sample_num}.json'
        )
        cache_fn = (
            f"{dataset_name}-{self.sampler_name}-"
            f"anchor_sample_num: {self.anchor_sample_num}:{self.candidate_num}"
            f"{'-' if other_info else '' + other_info}.json"
        )
        self.cache_file = os.path.join(self.cache_dir, cache_fn)
        self.index_ds_len = index_ds_len
        if anchor_idx_list is None:
            self.anchor_idx_list = self.sample_anchor_set()
        else:
            self.anchor_idx_list = anchor_idx_list

    def __call__(self, train_ds) -> Any:
        total_data = {}
        total_data['anchor_set'] = self.anchor_idx_list
        data = self.load_cache_file()
        if data is not None:
            total_data['candidate_set'] = data
            return total_data
        data = self.sample(self.anchor_idx_list, train_ds)
        self.save_cache_file(data)
        total_data['candidate_set'] = data
        return total_data

    def sample(self, *args, **kwargs):
        raise NotImplemented

    def load_cache_file(self):
        if not os.path.exists(self.cache_file) or self.overwrite:
            logger.info(
                f'the candidate set cache {self.cache_file} not exists or set overwrite mode. (overwrite: {self.overwrite})'
            )
            return
        else:
            logger.info(
                f'the candidate set cache {self.cache_file} exists, reloding...'
            )
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            data = {int(k): v for k, v in data.items()}
            return data

    def save_cache_file(self, data):
        with open(self.cache_file, 'w') as f:
            logger.info(f'save the candidate data to {self.cache_file}')
            json.dump(data, f)

    def sample_anchor_set(self):
        logger.info(self.anchor_set_cache_fn)
        if os.path.exists(self.anchor_set_cache_fn) and not self.overwrite:
            logger.info('the anchor_set_cache_filename exists, loding...')
            anchor_idx_list = json.load(open(self.anchor_set_cache_fn, 'r'))
        else:
            logger.info(
                f'the anchor set cache {self.anchor_set_cache_fn} not exists or set overwrite mode. (overwrite: {self.overwrite})'
            )
            anchor_idx_list = random.sample(
                range(0, self.index_ds_len), self.anchor_sample_num
            )
            with open(self.anchor_set_cache_fn, 'w') as f:
                logger.info(f'save to {self.anchor_set_cache_fn}...')
                json.dump(anchor_idx_list, f)
        return anchor_idx_list
