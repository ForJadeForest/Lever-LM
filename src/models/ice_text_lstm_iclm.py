import torch
from torch import nn
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

from .base_iclm import BaseICLM


class ICETextLSTMICLM(nn.Moduel):
    def __init__(
        self,
        emb_dim,
        num_layers,
        dropout,
        index_ds_size,
        clip_name="openai/clip-vit-base-patch32",
        freeze_prefix_list=None,
        adpter=False,
    ):
        super().__init__()
        vocab_size = index_ds_size + 3
        self.vocab_embedding = nn.Embedding(vocab_size, embedding_dim=emb_dim)
        self.lm_model = nn.GRU(
            input_size=emb_dim,
            hidden_size=emb_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, vocab_size),
        )
        self.img_model = CLIPVisionModelWithProjection.from_pretrained(clip_name)

        self._adpter = adpter
        if self._adpter:
            self.img_adpter = nn.Sequential(
                nn.Linear(self.img_model.config.projection_dim, emb_dim * 4),
                nn.ReLU(),
                nn.Linear(emb_dim * 4, emb_dim),
            )

        self.sen_model = CLIPTextModelWithProjection.from_pretrained(clip_name)
        if self._adpter:
            self.sen_adpter = nn.Sequential(
                nn.Linear(self.sen_model.config.projection_dim, emb_dim * 4),
                nn.ReLU(),
                nn.Linear(emb_dim * 4, emb_dim),
            )
        self.criterion = nn.CrossEntropyLoss()
        self.freeze_prefix(freeze_prefix_list)

    def forward(self, img_input, ice_input, ice_seq_idx):
        image_embeds = self.img_model(img_input)['image_embeds']
        if self._adpter:
            image_embeds = self.img_adpter(image_embeds)
        inputs_embeds = self.vocab_embedding(ice_seq_idx)
        inputs_embeds[:, 1] += image_embeds

        if ice_input is None:
            logits, _ = self.lm_model(inputs_embeds)
            logits = self.proj(logits)
            return {'logits': logits}

        bs, ice_num, ice_seq_len = ice_input['input_ids'].shape

        ice_input['input_ids'] = ice_input['input_ids'].view(-1, ice_seq_len)
        ice_input['attention_mask'] = ice_input['attention_mask'].view(-1, ice_seq_len)
        ice_features = self.sen_model(
            input_ids=ice_input['input_ids'],
            attention_mask=ice_input['attention_mask'],
        )['text_embeds']

        if self._adpter:
            ice_features = self.sen_adpter(ice_features)
        ice_features = ice_features.view(bs, ice_num, -1)
        inputs_embeds[:, 2 : 2 + ice_num] += ice_features

        logits, _ = self.lm_model(inputs_embeds=inputs_embeds)
        logits = self.proj(logits)  # batch, len, vocab_dim
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = ice_seq_idx[..., 1:].contiguous()
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return {'logits': logits, 'loss': loss}

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
        img_input,
        init_ice_idx,
        shot_num,
        index_ds,
        processor,
        text_field,
        device,
        repetition_penalty=2.0,
    ):
        """
        Generate for one batch
        """
        ice_input = None
        ice_seq_idx = init_ice_idx
        sp_token_begin = len(index_ds)

        for _ in range(shot_num):
            out = self.forward(img_input, ice_input, ice_seq_idx).logits[:, -1, :]
            # set the sp token prob to 0

            out[:, sp_token_begin:] = -torch.inf
            for ice_idx in ice_seq_idx:
                out[:, ice_idx] /= repetition_penalty

            next_token_idx = torch.softmax(out, dim=-1).argmax(dim=-1)

            ice_seq_idx = torch.cat(
                [ice_seq_idx, torch.tensor([[next_token_idx]], device=device)],
                dim=1,
            )

            ice_text_list = [
                index_ds[i][text_field] for i in ice_seq_idx.tolist()[0][2:]
            ]
            ice_input = processor(
                text=ice_text_list, padding=True, return_tensors='pt'
            ).to(device)

            # shape: 1, ice_num ,seq_len
            ice_input = {k: v.unsqueeze(dim=0) for k, v in ice_input.items()}

        return ice_seq_idx.detach().cpu().tolist()
