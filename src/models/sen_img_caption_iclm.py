import torch
from torch import nn
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    GPT2Config,
    GPT2LMHeadModel,
)


class SenImgEncodeCaptionICLM(nn.Module):
    def __init__(self, lm_config, clip_name="openai/clip-vit-base-patch32"):
        super().__init__()
        conifg = GPT2Config(
            vocab_size=lm_config.vocab_size,
            n_embd=lm_config.n_embd,
            n_head=lm_config.n_head,
            n_layer=lm_config.n_layer,
        )
        self.lm_model = GPT2LMHeadModel(conifg)
        self.sen_model = CLIPTextModelWithProjection.from_pretrained(clip_name)
        self.img_model = CLIPVisionModelWithProjection.from_pretrained(clip_name)

    def forward(self, img_input, ice_input=None, ice_seq_idx=None):
        if ice_input is not None:
            bs, ice_num, ice_seq_len = ice_input['input_ids'].shape

            ice_input['input_ids'] = ice_input['input_ids'].view(-1, ice_seq_len)
            ice_input['attention_mask'] = ice_input['attention_mask'].view(
                -1, ice_seq_len
            )
            ice_features = self.sen_model(
                input_ids=ice_input['input_ids'],
                attention_mask=ice_input['attention_mask'],
            )['text_embeds']
            ice_features = ice_features.view(bs, ice_num, -1)

            x_features = self.img_model(img_input)['image_embeds']

            lm_emb_input = torch.cat((x_features.unsqueeze(1), ice_features), dim=1)

        else:
            x_features = self.img_model(img_input)['image_embeds']
            lm_emb_input = x_features.unsqueeze(0)

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

        lm_output = self.lm_model(inputs_embeds=lm_emb_input, labels=labels)

        return lm_output
