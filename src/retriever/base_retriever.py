from typing import List


class BaseRetriever:
    def __init__(self, index_ds, test_ds) -> None:
        self.index_ds = index_ds
        self.test_ds = test_ds
        
    def retrieve(self, shot_num)-> List[List[int]]:
        raise NotImplemented
        