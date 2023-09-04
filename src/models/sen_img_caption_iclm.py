import torch
from torch import nn
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    GPT2Config,
    GPT2LMHeadModel,
)


class SenImgEncodeCaptionICLM(nn.Module):
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
        bs, ice_num, ice_seq_len = ice_input['input_ids'].shape
        dataset_embedding = self.lm_model.transformer.wte(ice_seq_idx)

        ice_input['input_ids'] = ice_input['input_ids'].view(-1, ice_seq_len)
        ice_input['attention_mask'] = ice_input['attention_mask'].view(-1, ice_seq_len)
        ice_features = self.sen_model(
            input_ids=ice_input['input_ids'],
            attention_mask=ice_input['attention_mask'],
        )['text_embeds']
        ice_features = ice_features.view(bs, ice_num, -1)
        img_features = self.img_model(img_input)['image_embeds']
        dataset_embedding[:, 1] += img_features
        dataset_embedding[:, 2:-1] += ice_features
        
        output = self.lm_model(inputs_embeds=dataset_embedding, labels=ice_seq_idx)
        return output
        

    @torch.inference_mode()
    def generation(self, img_input, ice_input, **kwargs):
        image_embedding = self.img_model(img_input)['image_embeds']
        dataset_embedding = self.lm_model.transformer.wte(ice_input)
        dataset_embedding[:, 1] += image_embedding
        generated_ids = self.lm_model.generate(
            input_ids=ice_input, inputs_embeds=dataset_embedding, **kwargs
        )

        return generated_ids.cpu().detach().tolist()
