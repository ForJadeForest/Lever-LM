import torch
from torch import nn
from transformers import CLIPVisionModelWithProjection, GPT2Config, GPT2LMHeadModel


class IdxBaseCaptionICLM(nn.Module):
    def __init__(
        self,
        lm_config,
        index_ds_size,
        init_with_vlm_logits=False,
        emb_dim=None,
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

        if init_with_vlm_logits:
            self.lm_model.transformer.wte = nn.Embedding(vocab_size, emb_dim)
            self.emb_proj = nn.Linear(emb_dim, lm_config['n_embd'])
            self.img_proj = nn.Linear(
                self.img_model.config.projection_dim, lm_config['n_embd']
            )
        else:
            self.emb_proj = nn.Identity(lm_config['n_embd'])
            self.img_proj = nn.Identity(self.img_model.config.projection_dim)

    def forward(self, img_input, ice_input, ice_seq_idx=None):
        image_embedding = self.img_model(img_input)['image_embeds']
        image_embedding = self.img_proj(image_embedding)

        dataset_embedding = self.lm_model.transformer.wte(ice_input)
        dataset_embedding = self.emb_proj(dataset_embedding)
        dataset_embedding[:, 1] += image_embedding
        inputs_embeds = dataset_embedding

        labels = ice_seq_idx
        # if labels is not None:
        #     labels[:, 1] = -100

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

    @torch.inference_mode()
    def generation(self, img_input, ice_input, **kwargs):
        image_embedding = self.img_model(img_input)['image_embeds']
        image_embedding = self.img_proj(image_embedding)

        dataset_embedding = self.lm_model.transformer.wte(ice_input)
        dataset_embedding = self.emb_proj(dataset_embedding)
        dataset_embedding[:, 1] += image_embedding
        inputs_embeds = dataset_embedding
        generated_ids = self.lm_model.generate(
            input_ids=ice_input, inputs_embeds=inputs_embeds, **kwargs
        )

        return generated_ids.cpu().detach().tolist()


if __name__ == '__main__':
    from transformers import AutoProcessor

    from datasets import load_dataset

    lm_config = {
        'n_embd': 512,
        'n_head': 8,
        'n_layer': 2,
    }
    model = IdxBaseCaptionICLM(lm_config, 118287)
    model.load_state_dict(
        torch.load('result/model_cpk/idx_base_iclm_2shot2/last-val_loss:5.5326.pth')[
            'model'
        ]
    )
    model = model.to('cuda:6')
    model.eval()
    ds = load_dataset("imagefolder", data_dir='/data/share/pyz/data/mscoco/mscoco2017')
    temp = ds['validation'][399]
    img = temp['image']
    img_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    img = img_processor(images=img, return_tensors='pt').to('cuda:6')['pixel_values']
    model.generation(img)
