import os

import torch
from loguru import logger

from icd_lm.utils import encode_image, recall_sim_feature

from .base_sampler import BaseSampler


class ImgSimSampler(BaseSampler):
    def __init__(
        self,
        candidate_num,
        sampler_name,
        anchor_sample_num,
        index_ds_len,
        cache_dir,
        dataset_name,
        overwrite,
        clip_model_name,
        feature_cache_filename,
        img_field_name,
        device,
        candidate_set_encode_bs,
        anchor_idx_list=None,
    ):
        super().__init__(
            candidate_num=candidate_num,
            sampler_name=sampler_name,
            dataset_name=dataset_name,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            cache_dir=cache_dir,
            overwrite=overwrite,
            other_info=feature_cache_filename.replace('openai/', ''),
            anchor_idx_list=anchor_idx_list,
        )
        self.clip_model_name = clip_model_name
        self.feature_cache_filename = feature_cache_filename.replace('openai/', '')
        self.feature_cache = os.path.join(self.cache_dir, self.feature_cache_filename)
        self.img_field_name = img_field_name
        self.device = device
        self.bs = candidate_set_encode_bs

    def sample(self, anchor_set, train_ds):
        if os.path.exists(self.feature_cache):
            logger.info(f'feature cache {self.feature_cache} exists, loding...')
            features = torch.load(self.feature_cache)
        else:
            features = encode_image(
                train_ds,
                self.img_field_name,
                self.device,
                self.clip_model_name,
                self.bs,
            )
            logger.info(f'saving the features cache in {self.feature_cache} ...')
            torch.save(features, self.feature_cache)
        test_feature = features[anchor_set]
        _, sim_sample_idx = recall_sim_feature(
            test_feature, features, top_k=self.candidate_num + 1
        )
        sim_sample_idx = sim_sample_idx[:, 1:].tolist()
        candidate_set_idx = {idx: cand for idx, cand in zip(anchor_set, sim_sample_idx)}
        return candidate_set_idx
