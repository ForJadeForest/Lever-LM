import torch
from torch import nn
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    GPT2Config,
    GPT2LMHeadModel,
)

from .base_iclm import BaseICLM


class GPT2ICLM(BaseICLM):
    def __init__(
        self,
        lm_config,
        index_ds_size: int,
        clip_name: str = "openai/clip-vit-base-patch32",
        adpter: bool = False,
        norm: bool = False,
        freeze_prefix_list: list = None,
        query_encoding_flag: list = None,
        ice_encoding_flag: list = None,
    ):
        super().__init__(
            adpter,
            norm,
            query_encoding_flag,
            ice_encoding_flag,
        )
        vocab_size = index_ds_size + 4
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=lm_config['n_embd'],
            n_head=lm_config['n_head'],
            n_layer=lm_config['n_layer'],
            eos_token_id=index_ds_size,
            bos_token_id=index_ds_size + 1,
            pad_token_id=index_ds_size + 3,
        )
        self.lm_model = GPT2LMHeadModel(config)
        need_encoder = set(self.query_encoding_flag + self.ice_encoding_flag)
        if 'image' in need_encoder:
            self.img_model = CLIPVisionModelWithProjection.from_pretrained(clip_name)
        if 'text' in need_encoder:
            self.sen_model = CLIPTextModelWithProjection.from_pretrained(clip_name)

        if self._adpter:
            if 'image' in need_encoder:
                self.img_adpter = nn.Sequential(
                    nn.Linear(
                        self.img_model.config.projection_dim, lm_config.n_embd * 4
                    ),
                    nn.ReLU(),
                    nn.Linear(lm_config.n_embd * 4, lm_config.n_embd),
                )
            if 'text' in need_encoder:
                self.sen_adpter = nn.Sequential(
                    nn.Linear(
                        self.sen_model.config.projection_dim, lm_config.n_embd * 4
                    ),
                    nn.ReLU(),
                    nn.Linear(lm_config.n_embd * 4, lm_config.n_embd),
                )
        self.freeze_prefix(freeze_prefix_list)

    def forward(self, query_input, ice_input, ice_seq_idx):
        text_embeds = image_embeds = None
        pad_token_id = self.lm_model.config.pad_token_id
        inputs_embeds = self.lm_model.transformer.wte(ice_seq_idx)

        # add query feature
        if 'image' in self.query_encoding_flag:
            image_embeds = self.img_model(query_input['pixel_values'])['image_embeds']
            if self._adpter:
                image_embeds = self.img_adpter(image_embeds)
            if self._norm:
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            inputs_embeds[:, 1] = inputs_embeds[:, 1] + image_embeds
        if 'text' in self.query_encoding_flag:
            text_embeds = self.sen_model(
                input_ids=query_input['input_ids'],
                attention_mask=query_input['attention_mask'],
            )['text_embeds']
            if self._adpter:
                text_embeds = self.sen_adpter(text_embeds)
            if self._norm:
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            inputs_embeds[:, 1] = inputs_embeds[:, 1] + text_embeds

        # add ice feature
        if ice_input is None:
            lm_output = self.lm_model(inputs_embeds=inputs_embeds)
            return lm_output
        if 'text' in self.ice_encoding_flag:
            ice_text_features = self.sen_model(
                input_ids=ice_input['input_ids'],
                attention_mask=ice_input['attention_mask'],
            )['text_embeds']
            if self._adpter:
                ice_text_features = self.sen_adpter(ice_text_features)
            if self._norm:
                ice_text_features = ice_text_features / ice_text_features.norm(
                    dim=-1, keepdim=True
                )

            split_sizes = ice_input['ice_num_list'].cpu().tolist()
            ice_text_feature_splits = torch.split(ice_text_features, split_sizes, dim=0)
            for i, ice_num in enumerate(ice_input['ice_num_list']):
                inputs_embeds[i, 2 : 2 + ice_num] = (
                    inputs_embeds[i, 2 : 2 + ice_num] + ice_text_feature_splits[i]
                )

        if 'image' in self.ice_encoding_flag:
            ice_img_features = self.img_model(ice_input['pixel_values'])['image_embeds']

            if self._adpter:
                ice_img_features = self.img_adpter(ice_img_features)
            if self._norm:
                ice_img_features = ice_img_features / ice_img_features.norm(
                    dim=-1, keepdim=True
                )
            split_sizes = ice_input['ice_num_list'].cpu().tolist()
            ice_image_feature_splits = torch.split(ice_img_features, split_sizes, dim=0)
            for i, ice_num in enumerate(ice_input['ice_num_list']):
                inputs_embeds[i, 2 : 2 + ice_num] = (
                    inputs_embeds[i, 2 : 2 + ice_num] + ice_image_feature_splits[i]
                )
        ice_seq_idx = torch.where(
            ice_seq_idx == pad_token_id,
            torch.tensor(-100, device=ice_seq_idx.device),
            ice_seq_idx,
        )

        output = self.lm_model(inputs_embeds=inputs_embeds, labels=ice_seq_idx)
        return output

    @torch.inference_mode()
    def generation(
        self,
        query_input,
        init_ice_idx,
        shot_num,
        index_ds,
        processor,
        device,
        ice_image_field,
        ice_text_field,
        repetition_penalty=2.0,
    ):
        """
        Generate for one batch
        """
        ice_input = None
        ice_seq_idx = init_ice_idx
        sp_token_begin = len(index_ds)
        bs = len(ice_seq_idx)

        for s_n in range(shot_num):
            out = self.forward(query_input, ice_input, ice_seq_idx)["logits"][:, -1, :]
            # set the sp token prob to 0
            out[:, sp_token_begin:] = -torch.inf
            for ice_idx in ice_seq_idx:
                out[:, ice_idx] /= repetition_penalty

            next_token_idx = torch.softmax(out, dim=-1).argmax(dim=-1)  # bs,

            ice_seq_idx = torch.cat(
                [ice_seq_idx, next_token_idx.unsqueeze(dim=1)], dim=1
            )
            ice_text_list = ice_img_list = None
            if 'text' in self.ice_encoding_flag:
                ice_text_list = [
                    index_ds[idx][ice_text_field]
                    for i in range(bs)
                    for idx in ice_seq_idx.tolist()[i][2:]
                ]
            if 'image' in self.ice_encoding_flag:
                ice_img_list = [
                    index_ds[idx][ice_image_field]
                    for i in range(bs)
                    for idx in ice_seq_idx.tolist()[i][2:]
                ]
            if ice_text_list or ice_img_list:
                ice_input = processor(
                    text=ice_text_list,
                    images=ice_img_list,
                    padding=True,
                    return_tensors='pt',
                ).to(device)
                ice_num_list = [s_n for _ in range(bs)]
                ice_input['ice_num_list'] = torch.tensor(ice_num_list).to(device)
        return ice_seq_idx.detach().cpu().tolist()
