import torch
from torch import nn
from transformers import CLIPVisionModelWithProjection, GPT2Config, GPT2LMHeadModel


class BaseICLM(nn.Module):
    def __init__(
        self,
        lm_config,
        index_ds_size,
        clip_name="openai/clip-vit-base-patch32",
    ) -> None:
        super().__init__()
        vocab_size = index_ds_size + 3
        conifg = GPT2Config(
            vocab_size=vocab_size,
            n_embd=lm_config['n_embd'],
            n_head=lm_config['n_head'],
            n_layer=lm_config['n_layer'],
            eos_token_id=vocab_size,
            bos_token_id=vocab_size + 1,
        )
        self.lm_model = GPT2LMHeadModel(conifg)
        self.img_model = CLIPVisionModelWithProjection.from_pretrained(clip_name)

    def forward(self, img_input, ice_input):
        image_embeds = self.img_model(img_input)['image_embeds']
        dataset_embeds = self.lm_model.transformer.wte(ice_input)
        dataset_embeds[:, 1] += image_embeds
        return dataset_embeds

    @torch.inference_mode()
    def generation(self, *args, **kwargs):
        raise NotImplemented()
