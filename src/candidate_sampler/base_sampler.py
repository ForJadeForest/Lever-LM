import json
import os
from typing import Any

from loguru import logger


class BaseSampler:
    def __init__(
        self, candidate_num, sampler_name, cache_dir, overwrite, dataset_name, other_info=''
    ) -> None:
        self.candidate_num = candidate_num
        self.sampler_name = sampler_name
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.overwrite = overwrite
        self.cache_file = os.path.join(
            self.cache_dir,
            f"{dataset_name}-{self.sampler_name}-{other_info}:{self.candidate_num}.json",
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        data = self.load_cache_file()
        if data is not None:
            return data
        data = self.sample(*args, **kwargs)
        self.save_cache_file(data)
        return data

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
