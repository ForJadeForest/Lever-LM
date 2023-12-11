import numpy as np
from loguru import logger
from tqdm import trange


class RandRetriever:
    def __init__(self, index_ds, test_ds, seed=42, fixed=False):
        self.index_ds = index_ds
        self.test_ds = test_ds
        self.seed = seed
        self.fixed = fixed

    def retrieve(self, shot_num):
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        if self.fixed:
            idx_list = np.random.choice(num_idx, shot_num, replace=False).tolist()
            return [idx_list for _ in range(len(self.test_ds))]

        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for _ in trange(len(self.test_ds)):
            idx_list = np.random.choice(num_idx, shot_num, replace=False).tolist()
            rtr_idx_list.append(idx_list)
        return rtr_idx_list
