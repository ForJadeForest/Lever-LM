import torch
from torch import nn


class BaseLeverLM(nn.Module):
    def __init__(
        self,
        adapter: bool = False,
        norm: bool = False,
        query_encoding_flag: list = None,
        icd_encoding_flag: list = None,
    ) -> None:
        super().__init__()
        if query_encoding_flag is None:
            query_encoding_flag = []

        self._adapter = adapter
        self._norm = norm
        self.query_encoding_flag = query_encoding_flag
        self.icd_encoding_flag = icd_encoding_flag

    def forward(*args, **kwargs):
        raise NotImplemented()

    def freeze_prefix(self, freeze_prefix_list):
        if freeze_prefix_list is None:
            return
        for n, p in self.named_parameters():
            for prefix in freeze_prefix_list:
                if n.startswith(prefix):
                    p.requires_grad = False

    @torch.inference_mode()
    def generation(self, *args, **kwargs):
        raise NotImplemented()
