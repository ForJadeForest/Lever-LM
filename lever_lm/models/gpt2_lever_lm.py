import torch
from torch import nn
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    GPT2Config,
    GPT2LMHeadModel,
)

from .base_lever_lm import BaseLeverLM


class GPT2LeverLM(BaseLeverLM):
    def __init__(
        self,
        lm_config,
        index_ds_size: int,
        clip_name: str = "openai/clip-vit-base-patch32",
        adapter: bool = False,
        norm: bool = False,
        freeze_prefix_list: list = None,
        query_encoding_flag: list = None,
        icd_encoding_flag: list = None,
    ):
        super().__init__(
            adapter,
            norm,
            query_encoding_flag,
            icd_encoding_flag,
        )
        vocab_size = index_ds_size + 3
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=lm_config["n_embd"],
            n_head=lm_config["n_head"],
            n_layer=lm_config["n_layer"],
            eos_token_id=vocab_size,
            bos_token_id=vocab_size + 1,
        )
        self.lm_model = GPT2LMHeadModel(config)
        need_encoder = set(self.query_encoding_flag + self.icd_encoding_flag)
        if "image" in need_encoder:
            self.img_model = CLIPVisionModelWithProjection.from_pretrained(clip_name)
        if "text" in need_encoder:
            self.sen_model = CLIPTextModelWithProjection.from_pretrained(clip_name)

        if self._adapter:
            if "image" in need_encoder:
                self.img_adapter = nn.Sequential(
                    nn.Linear(
                        self.img_model.config.projection_dim, lm_config.n_embd * 4
                    ),
                    nn.ReLU(),
                    nn.Linear(lm_config.n_embd * 4, lm_config.n_embd),
                )
            if "text" in need_encoder:
                self.sen_adapter = nn.Sequential(
                    nn.Linear(
                        self.sen_model.config.projection_dim, lm_config.n_embd * 4
                    ),
                    nn.ReLU(),
                    nn.Linear(lm_config.n_embd * 4, lm_config.n_embd),
                )
        self.freeze_prefix(freeze_prefix_list)

    def forward(self, query_input, icd_input, icd_seq_idx):
        text_embeds = image_embeds = None
        inputs_embeds = self.lm_model.transformer.wte(icd_seq_idx)

        # add query feature
        if "image" in self.query_encoding_flag:
            image_embeds = self.img_model(query_input["pixel_values"])["image_embeds"]
            if self._adapter:
                image_embeds = self.img_adapter(image_embeds)
            if self._norm:
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            inputs_embeds[:, 1] += image_embeds
        if "text" in self.query_encoding_flag:
            text_embeds = self.sen_model(
                input_ids=query_input["input_ids"],
                attention_mask=query_input["attention_mask"],
            )["text_embeds"]
            if self._adapter:
                text_embeds = self.sen_adapter(text_embeds)
            if self._norm:
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            inputs_embeds[:, 1] += text_embeds

        # add icd feature
        if icd_input is None:
            lm_output = self.lm_model(inputs_embeds=inputs_embeds)
            return lm_output
        if "text" in self.icd_encoding_flag:
            bs, icd_num, icd_seq_len = icd_input["input_ids"].shape
            icd_input["input_ids"] = icd_input["input_ids"].view(-1, icd_seq_len)
            icd_input["attention_mask"] = icd_input["attention_mask"].view(
                -1, icd_seq_len
            )

            icd_text_features = self.sen_model(
                input_ids=icd_input["input_ids"],
                attention_mask=icd_input["attention_mask"],
            )["text_embeds"]
            if self._adapter:
                icd_text_features = self.sen_adapter(icd_text_features)
            if self._norm:
                icd_text_features = icd_text_features / icd_text_features.norm(
                    dim=-1, keepdim=True
                )
            icd_text_features = icd_text_features.view(bs, icd_num, -1)
            inputs_embeds[:, 2 : 2 + icd_num] += icd_text_features
        if "image" in self.icd_encoding_flag:
            bs, icd_num = icd_input["pixel_values"].shape[:2]
            img_shape = icd_input["pixel_values"].shape[-3:]
            icd_input["pixel_values"] = icd_input["pixel_values"].view(-1, *img_shape)
            icd_img_features = self.img_model(icd_input["pixel_values"])["image_embeds"]

            if self._adapter:
                icd_img_features = self.img_adapter(icd_img_features)
            if self._norm:
                icd_img_features = icd_img_features / icd_img_features.norm(
                    dim=-1, keepdim=True
                )
            icd_img_features = icd_img_features.view(bs, icd_num, -1)
            inputs_embeds[:, 2 : 2 + icd_num] += icd_img_features

        output = self.lm_model(inputs_embeds=inputs_embeds, labels=icd_seq_idx)
        return output

    @torch.inference_mode()
    def generation(
        self,
        query_input,
        init_icd_idx,
        shot_num,
        index_ds,
        processor,
        device,
        icd_image_field,
        icd_text_field,
    ):
        """
        Generate for one batch
        """
        icd_input = None
        icd_seq_idx = init_icd_idx
        sp_token_begin = len(index_ds)
        bs = len(icd_seq_idx)

        for s_n in range(shot_num):
            out = self.forward(query_input, icd_input, icd_seq_idx)["logits"][:, -1, :]
            # set the sp token prob to 0
            out[:, sp_token_begin:] = -torch.inf
            for icd_idx in icd_seq_idx:
                out[:, icd_idx] = -torch.inf

            next_token_idx = torch.softmax(out, dim=-1).argmax(dim=-1)  # bs, 1

            icd_seq_idx = torch.cat(
                [icd_seq_idx, next_token_idx.unsqueeze(dim=1)], dim=1
            )
            icd_text_list = icd_img_list = None
            if "text" in self.icd_encoding_flag:
                icd_text_list = [
                    index_ds[idx][icd_text_field]
                    for i in range(bs)
                    for idx in icd_seq_idx.tolist()[i][2:]
                ]
            if "image" in self.icd_encoding_flag:
                icd_img_list = [
                    index_ds[idx][icd_image_field]
                    for i in range(bs)
                    for idx in icd_seq_idx.tolist()[i][2:]
                ]
            if icd_text_list or icd_img_list:
                flatten_icd_input = processor(
                    text=icd_text_list,
                    images=icd_img_list,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                icd_input = {}
                for k in flatten_icd_input:
                    other_dim = flatten_icd_input[k].shape[1:]
                    icd_input[k] = flatten_icd_input[k].view(bs, s_n + 1, *other_dim)
        return icd_seq_idx.detach().cpu().tolist()
