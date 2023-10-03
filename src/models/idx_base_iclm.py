import torch

from .base_iclm import BaseICLM


class IdxBaseICLM(BaseICLM):
    def __init__(
        self,
        lm_config,
        index_ds_size,
        clip_name="openai/clip-vit-base-patch32",
        adpter=False,
        norm=False
    ) -> None:
        super().__init__(lm_config, index_ds_size, clip_name, adpter, norm)

    def forward(self, img_input, ice_seq_idx):
        inputs_embeds = super().forward(img_input, ice_seq_idx)
        lm_output = self.lm_model(
            inputs_embeds=inputs_embeds,
            labels=ice_seq_idx,
        )
        return lm_output

    @torch.inference_mode()
    def generation(self, img_input, init_ice_idx, **kwargs):
        inputs_embeds = super().forward(img_input, init_ice_idx)
        generated_ids = self.lm_model.generate(
            input_ids=init_ice_idx, inputs_embeds=inputs_embeds, **kwargs
        )

        return generated_ids.cpu().detach().tolist()
