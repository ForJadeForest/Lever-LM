import os
import random

import torch
from loguru import logger

from src.utils import encode_text, recall_sim_feature

from .base_sampler import BaseSampler
from .img_sim_sampler import ImgSimSampler
from .random_sampler import RandSampler
from .text_sim_sampler import TextSimSampler


class MixSimSampler(BaseSampler):
    def __init__(
        self,
        candidate_num,
        dataset_name,
        sampler_name,
        anchor_sample_num,
        index_ds_len,
        cache_dir,
        overwrite,
        clip_model_name,
        feature_cache_filename,
        text_field_name,
        img_field_name,
        device,
        candidate_set_encode_bs,
        sampler_ratio: dict,
    ):
        self.sampler_ratio = sampler_ratio
        self.sampler_candidate_num = {}
        for i, (n, r) in enumerate(self.sampler_ratio.items()):
            if i == len(self.sampler_ratio) - 1:
                self.sampler_candidate_num[n] = candidate_num - sum(
                    [v for v in self.sampler_candidate_num.values()]
                )
            else:
                self.sampler_candidate_num[n] = int(r * candidate_num)
        other_info = (
            f'Rand:{self.sampler_candidate_num["RandSampler"]}-'
            f'Text:{self.sampler_candidate_num["TextSimSampler"]}-'
            f'Img:{self.sampler_candidate_num["ImgSimSampler"]}'
        )
        super().__init__(
            candidate_num=candidate_num,
            dataset_name=dataset_name,
            sampler_name=sampler_name,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            cache_dir=cache_dir,
            overwrite=overwrite,
            other_info=other_info,
        )
        self.rand_sampler = RandSampler(
            candidate_num=self.sampler_candidate_num['RandSampler'],
            sampler_name=sampler_name,
            dataset_name=dataset_name,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            cache_dir=cache_dir,
            overwrite=overwrite,
            anchor_idx_list=self.anchor_idx_list,
        )
        self.text_sim_sampler = TextSimSampler(
            candidate_num=self.sampler_candidate_num['TextSimSampler'],
            sampler_name=None,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            overwrite=overwrite,
            clip_model_name=clip_model_name,
            feature_cache_filename=feature_cache_filename + '-TextFeatures.pth',
            text_field_name=text_field_name,
            device=device,
            candidate_set_encode_bs=candidate_set_encode_bs,
            anchor_idx_list=self.anchor_idx_list,
        )
        self.img_sim_sampler = ImgSimSampler(
            candidate_num=self.sampler_candidate_num['ImgSimSampler'],
            sampler_name=None,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            cache_dir=cache_dir,
            overwrite=overwrite,
            dataset_name=dataset_name,
            clip_model_name=clip_model_name,
            feature_cache_filename=feature_cache_filename + '-ImgFeatures.pth',
            img_field_name=img_field_name,
            device=device,
            candidate_set_encode_bs=candidate_set_encode_bs,
            anchor_idx_list=self.anchor_idx_list,
        )

    @torch.inference_mode()
    def sample(self, train_ds):
        final_res = {k: [] for k in self.anchor_idx_list}
        rand_res = self.rand_sampler.sample(train_ds)
        img_sim_res = self.img_sim_sampler.sample(train_ds)
        text_sim_res = self.text_sim_sampler.sample(train_ds)
        for k in self.anchor_idx_list:
            final_res[k].extend(rand_res[k])
            final_res[k].extend(img_sim_res[k])
            final_res[k].extend(text_sim_res[k])

        return final_res
