import torch
from torch import nn
from transformers import GPT2LMHeadModel


class BaseCaptionICLM(nn.Module):
    def __init__(self, lm_config, emb_dim) -> None:
        super().__init__()
        self.lm_model = GPT2LMHeadModel(lm_config)
        self.lm_model.transformer.wte = nn.Embedding(lm_config.vocab_size, emb_dim)
        self.emb_proj = nn.Linear(emb_dim, lm_config.n_embd)

    def forward(
        self, test_sample_embedding, seq_input_ids=None, seq_attention_mask=None
    ):
        test_sample_embedding = self.emb_proj(test_sample_embedding)
        bs = len(test_sample_embedding)
        if seq_input_ids is None:
            inputs_embeds = test_sample_embedding.unsqueeze(dim=1)
            lm_output = self.lm_model(inputs_embeds=inputs_embeds, attention_mask=None)
        else:
            dataset_embedding = self.lm_model.transformer.wte(seq_input_ids)
            dataset_embedding = self.emb_proj(dataset_embedding)
            if seq_attention_mask is not None:
                padding_mask = torch.ones(
                    (bs, 1), device=seq_input_ids.device, dtype=torch.int32
                )
                seq_attention_mask = torch.cat(
                    [padding_mask, seq_attention_mask], dim=1
                )

            inputs_embeds = torch.cat(
                [test_sample_embedding.unsqueeze(dim=1), dataset_embedding], dim=1
            )
            padding_labels = (
                torch.ones(
                    (len(seq_input_ids), 1),
                    device=inputs_embeds.device,
                    dtype=torch.int32,
                )
                * -100
            )
            labels = torch.cat([padding_labels, seq_input_ids], dim=1)

            lm_output = self.lm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=seq_attention_mask,
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
