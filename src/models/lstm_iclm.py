import torch
from torch import nn
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from .base_lever_lm import BaseICLM


class LSTMICLM(BaseICLM):
    def __init__(
        self,
        emb_dim,
        num_layers,
        dropout,
        index_ds_size,
        clip_name="openai/clip-vit-base-patch32",
        adpter=False,
        norm=False,
        freeze_prefix_list=None,
        query_encoding_flag: list = None,
        ice_encoding_flag: list = None,
    ):
        super().__init__(
            adpter,
            norm,
            query_encoding_flag,
            ice_encoding_flag,
        )
        vocab_size = index_ds_size + 3
        self.vocab_embedding = nn.Embedding(vocab_size, embedding_dim=emb_dim)
        self.lm_model = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, vocab_size),
        )

        need_encoder = set(self.query_encoding_flag + self.ice_encoding_flag)
        if "image" in need_encoder:
            self.img_model = CLIPVisionModelWithProjection.from_pretrained(clip_name)
        if "text" in need_encoder:
            self.sen_model = CLIPTextModelWithProjection.from_pretrained(clip_name)

        if self._adpter:
            if "image" in need_encoder:
                self.img_adpter = nn.Sequential(
                    nn.Linear(self.img_model.config.projection_dim, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                )
            if "text" in need_encoder:
                self.sen_adpter = nn.Sequential(
                    nn.Linear(self.sen_model.config.projection_dim, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, emb_dim),
                )
        self.criterion = nn.CrossEntropyLoss()
        self.freeze_prefix(freeze_prefix_list)

    def forward(self, query_input, ice_input, ice_seq_idx):
        text_embeds = image_embeds = None
        inputs_embeds = self.vocab_embedding(ice_seq_idx)

        # add query feature
        if "image" in self.query_encoding_flag:
            image_embeds = self.img_model(query_input["pixel_values"])["image_embeds"]
            if self._adpter:
                image_embeds = self.img_adpter(image_embeds)
            if self._norm:
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            inputs_embeds[:, 1] += image_embeds
        if "text" in self.query_encoding_flag:
            text_embeds = self.sen_model(
                input_ids=query_input["input_ids"],
                attention_mask=query_input["attention_mask"],
            )["text_embeds"]
            if self._adpter:
                text_embeds = self.sen_adpter(text_embeds)
            if self._norm:
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            inputs_embeds[:, 1] += text_embeds

        # add ice feature
        if ice_input is None:
            logits, _ = self.lm_model(inputs_embeds)
            logits = self.proj(logits)
            return {"logits": logits}
        if "text" in self.ice_encoding_flag:
            bs, ice_num, ice_seq_len = ice_input["input_ids"].shape
            ice_input["input_ids"] = ice_input["input_ids"].view(-1, ice_seq_len)
            ice_input["attention_mask"] = ice_input["attention_mask"].view(
                -1, ice_seq_len
            )

            ice_text_features = self.sen_model(
                input_ids=ice_input["input_ids"],
                attention_mask=ice_input["attention_mask"],
            )["text_embeds"]
            if self._adpter:
                ice_text_features = self.sen_adpter(ice_text_features)
            if self._norm:
                ice_text_features = ice_text_features / ice_text_features.norm(
                    dim=-1, keepdim=True
                )
            ice_text_features = ice_text_features.view(bs, ice_num, -1)
            inputs_embeds[:, 2 : 2 + ice_num] += ice_text_features
        if "image" in self.ice_encoding_flag:
            bs, ice_num = ice_input["pixel_values"].shape[:2]
            img_shape = ice_input["pixel_values"].shape[-3:]
            ice_input["pixel_values"] = ice_input["pixel_values"].view(-1, *img_shape)
            ice_img_features = self.img_model(ice_input["pixel_values"])["image_embeds"]

            if self._adpter:
                ice_img_features = self.img_adpter(ice_img_features)
            if self._norm:
                ice_img_features = ice_img_features / ice_img_features.norm(
                    dim=-1, keepdim=True
                )
            ice_img_features = ice_img_features.view(bs, ice_num, -1)
            inputs_embeds[:, 2 : 2 + ice_num] += ice_img_features

        logits, _ = self.lm_model(inputs_embeds)
        logits = self.proj(logits)  # batch, len, vocab_dim
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = ice_seq_idx[..., 1:].contiguous()
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return {"logits": logits, "loss": loss}

    def freeze_prefix(self, freeze_prefix_list):
        if freeze_prefix_list is None:
            return
        for n, p in self.named_parameters():
            for prefix in freeze_prefix_list:
                if n.startswith(prefix):
                    p.requires_grad = False

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
                out[:, ice_idx] = -torch.inf

            next_token_idx = torch.softmax(out, dim=-1).argmax(dim=-1)  # bs, 1

            ice_seq_idx = torch.cat(
                [ice_seq_idx, next_token_idx.unsqueeze(dim=1)], dim=1
            )
            ice_text_list = ice_img_list = None
            if "text" in self.ice_encoding_flag:
                ice_text_list = [
                    index_ds[idx][ice_text_field]
                    for i in range(bs)
                    for idx in ice_seq_idx.tolist()[i][2:]
                ]
            if "image" in self.ice_encoding_flag:
                ice_img_list = [
                    index_ds[idx][ice_image_field]
                    for i in range(bs)
                    for idx in ice_seq_idx.tolist()[i][2:]
                ]
            if ice_text_list or ice_img_list:
                flatten_ice_input = processor(
                    text=ice_text_list,
                    images=ice_img_list,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                ice_input = {}
                for k in flatten_ice_input:
                    other_dim = flatten_ice_input[k].shape[1:]
                    ice_input[k] = flatten_ice_input[k].view(bs, s_n + 1, *other_dim)
        return ice_seq_idx.detach().cpu().tolist()
