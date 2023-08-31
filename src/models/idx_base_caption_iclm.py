import torch
from torch import nn
from transformers import CLIPVisionModelWithProjection, GPT2Config, GPT2LMHeadModel


class IdxBaseCaptionICLM(nn.Module):
    def __init__(
        self,
        lm_config,
        init_with_vlm_logits=False,
        emb_dim=None,
        clip_name="openai/clip-vit-base-patch32",
    ) -> None:
        super().__init__()
        conifg = GPT2Config(
            vocab_size=lm_config.vocab_size,
            n_embd=lm_config.n_embd,
            n_head=lm_config.n_head,
            n_layer=lm_config.n_layer,
        )
        self.lm_model = GPT2LMHeadModel(conifg)
        self.img_model = CLIPVisionModelWithProjection.from_pretrained(clip_name)

        if init_with_vlm_logits:
            self.lm_model.transformer.wte = nn.Embedding(lm_config.vocab_size, emb_dim)
            self.emb_proj = nn.Linear(emb_dim, lm_config.n_embd)
            self.img_proj = nn.Linear(
                self.img_model.config.projection_dim, lm_config.n_embd
            )
        else:
            self.emb_proj = nn.Identity(lm_config.n_embd)
            self.img_proj = nn.Identity(self.img_model.config.projection_dim)

    def forward(self, img_input, ice_input=None, ice_seq_idx=None):
        image_embedding = self.img_model(**img_input)['image_embeds']
        image_embedding = self.img_proj(image_embedding)
        bs = len(image_embedding)

        if ice_input is None:
            inputs_embeds = image_embedding.unsqueeze(dim=1)
        else:
            dataset_embedding = self.lm_model.transformer.wte(ice_input)
            dataset_embedding = self.emb_proj(dataset_embedding)

            inputs_embeds = torch.cat(
                [image_embedding.unsqueeze(dim=1), dataset_embedding], dim=1
            )

        if ice_seq_idx is not None:
            padding_labels = (
                torch.ones(
                    (bs, 1),
                    device=ice_seq_idx.device,
                    dtype=torch.long,
                )
                * -100
            )
            labels = torch.cat([padding_labels, ice_seq_idx], dim=1)
        else:
            labels = None

        lm_output = self.lm_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
        )
        return lm_output

    @torch.inference_mode()
    def init_dataset_embedding(self, ds_emb_dict, freeze=True):
        dataset_emb = torch.stack(
            [ds_emb_dict[i] for i in range(len(ds_emb_dict))], dim=0
        )
        self.lm_model.transformer.wte.weight.data = dataset_emb
        if freeze:
            print('freeze dataset embedding')
            for p in self.lm_model.transformer.wte.parameters():
                p.requires_grad = False
