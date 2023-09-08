import torch
from torch import nn
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    GPT2Config,
    GPT2LMHeadModel,
)


class SenImgEncodeAddImageCaptionICLM(nn.Module):
    def __init__(
        self, lm_config, index_ds_size, clip_name="openai/clip-vit-base-patch32"
    ):
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
        self.sen_model = CLIPTextModelWithProjection.from_pretrained(clip_name)
        self.img_model = CLIPVisionModelWithProjection.from_pretrained(clip_name)

    def forward(self, img_input, ice_input, ice_seq_idx):
        if ice_input is None:
            image_embedding = self.img_model(img_input)['image_embeds']
            dataset_embedding = self.lm_model.transformer.wte(ice_seq_idx)
            dataset_embedding[:, 1] += image_embedding
            lm_output = self.lm_model(inputs_embeds=dataset_embedding)
            return lm_output

        bs, ice_num, ice_seq_len = ice_input['input_ids'].shape
        dataset_embedding = self.lm_model.transformer.wte(ice_seq_idx)

        ice_input['input_ids'] = ice_input['input_ids'].view(-1, ice_seq_len)
        ice_input['attention_mask'] = ice_input['attention_mask'].view(-1, ice_seq_len)
        img_shape = ice_input['pixel_values'].shape[-3:]
        ice_input['pixel_values'] = ice_input['pixel_values'].view(-1, *img_shape)

        ice_text_features = self.sen_model(
            input_ids=ice_input['input_ids'],
            attention_mask=ice_input['attention_mask'],
        )['text_embeds']

        ice_img_features = self.img_model(ice_input['pixel_values'])['image_embeds']
        ice_img_features = ice_img_features.view(bs, ice_num, -1)
        ice_text_features = ice_text_features.view(bs, ice_num, -1)

        ice_features = ice_text_features + ice_img_features

        img_features = self.img_model(img_input)['image_embeds']
        dataset_embedding[:, 1] += img_features
        dataset_embedding[:, 2 : 2 + ice_num] += ice_features

        output = self.lm_model(inputs_embeds=dataset_embedding, labels=ice_seq_idx)
        return output

    @torch.inference_mode()
    def generation(
        self, img_input, shot_num, coco_ds, processor, repetition_penalty=2.0
    ):
        """
        Generate for one batch
        """
        ice_input = None
        device = next(self.lm_model.parameters()).device
        ice_seq_idx = torch.tensor([[118288, 118289]]).to(device)

        for _ in range(shot_num):
            out = self.forward(img_input, ice_input, ice_seq_idx).logits[:, -1, :]
            # set the eos token prob to 0
            out[:, 118287:] = -torch.inf
            for ice_idx in ice_seq_idx:
                out[:, ice_idx] /= repetition_penalty

            next_token_idx = torch.softmax(out, dim=-1).argmax(dim=-1)

            ice_seq_idx = torch.cat(
                [ice_seq_idx, torch.tensor([[next_token_idx]], device=device)],
                dim=1,
            )

            ice_text_list = [
                coco_ds[i]['single_caption'] for i in ice_seq_idx.tolist()[0][2:]
            ]
            ice_img_list = [coco_ds[i]['image'] for i in ice_seq_idx.tolist()[0][2:]]
            ice_input = processor(
                text=ice_text_list,
                images=ice_img_list,
                padding=True,
                return_tensors='pt',
            ).to(device)
            ice_input = {k: v.unsqueeze(dim=0) for k, v in ice_input.items()}

        return ice_seq_idx.detach().cpu().tolist()
