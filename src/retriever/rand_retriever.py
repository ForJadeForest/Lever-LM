from typing import List

import datasets
import numpy as np
from loguru import logger
from tqdm import trange

from src.retriever.base_retriever import BaseRetriever


class RandRetriever(BaseRetriever):
    def __init__(
        self,
        index_ds: datasets.Dataset,
        test_ds: datasets.Dataset,
        seed: int = 42,
        fixed: bool = False,
    ):
        """Initialize the RandRetriever.

        Args:
            index_ds (datasets.Dataset): The dataset used for creating the index.
            test_ds (datasets.Dataset): The dataset used for testing.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.
            fixed (bool, optional): If True, use the same random indices for all test cases. Defaults to False.
        """
        super().__init__(index_ds, test_ds)
        self.seed = seed
        self.fixed = fixed

    def retrieve(self, shot_num: int) -> List[List[int]]:
        """Retrieve random indices from the index dataset.

        Args:
            shot_num (int): The number of random indices to retrieve for each test case.

        Returns:
            List[List[int]]: A list of lists containing the retrieved random indices. Each sublist corresponds to a test case.
        """

        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        if self.fixed:
            logger.info("Retrieving data for test set...")
            idx_list = np.random.choice(num_idx, shot_num, replace=False).tolist()
            return [idx_list for _ in range(len(self.test_ds))]

        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for _ in trange(len(self.test_ds)):
            idx_list = np.random.choice(num_idx, shot_num, replace=False).tolist()
            rtr_idx_list.append(idx_list)
        return rtr_idx_list
