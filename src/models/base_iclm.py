import torch
from torch import nn
from transformers import CLIPVisionModelWithProjection, GPT2Config, GPT2LMHeadModel


class BaseICLM(nn.Module):
    def __init__(
        self,
        lm_config,
        index_ds_size,
        clip_name="openai/clip-vit-base-patch32",
        freeze_prefix=None,
        adpter=False,
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
        self._adpter = adpter
        if self._adpter:
            self.img_adpter = nn.Sequential(
                nn.Linear(self.sen_model.config.projection_dim, lm_config.n_embd * 4),
                nn.Relu(),
                nn.Linear(lm_config.n_embd * 4, lm_config.n_embd),
            )
        self.freeze_prefix(freeze_prefix)

    def forward(self, img_input, ice_input):
        image_embeds = self.img_model(img_input)['image_embeds']
        if self.adpter:
            image_embeds = self.img_adpter(image_embeds)
        dataset_embeds = self.lm_model.transformer.wte(ice_input)
        dataset_embeds[:, 1] += image_embeds
        return dataset_embeds

    def freeze_prefix(self, prefix_list):
        for n, p in self.named_parameters():
            for prefix in prefix_list:
                if n.startswith(prefix):
                    print(f'freeze: {n}')
                    p.requires_grad = False

    @torch.inference_mode()
    def generation(self, *args, **kwargs):
        raise NotImplemented()
