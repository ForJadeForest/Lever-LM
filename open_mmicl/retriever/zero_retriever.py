from typing import List

import datasets

from open_mmicl.retriever.base_retriever import BaseRetriever


class ZeroRetriever(BaseRetriever):
    def __init__(
        self,
        index_ds: datasets.Dataset,
        test_ds: datasets.Dataset,
    ):
        """Initialize the RandRetriever.

        Args:
            index_ds (datasets.Dataset): The dataset used for creating the index.
            test_ds (datasets.Dataset): The dataset used for testing.
        """
        super().__init__(index_ds, test_ds)

    def retrieve(self, ice_num=0) -> List[List[int]]:
        """Retrieve random indices from the index dataset.

        Args:
            shot_num (int): The number of random indices to retrieve for each test case.

        Returns:
            List[List[int]]: A list of lists containing the retrieved random indices. Each sublist corresponds to a test case.
        """
        if ice_num != 0:
            raise ValueError(
                f'ZeroRetriever only support ice_num=0, but got {ice_num=}'
            )
        return [[] for _ in range(len(self.test_ds))]
