import torch
from transformers import CLIPTextModelWithProjection

from .base_iclm import BaseICLM


class ICEImageICLM(BaseICLM):
    def __init__(
        self,
        lm_config,
        index_ds_size,
        clip_name="openai/clip-vit-base-patch32",
        freeze_prefix_list=None,
        adpter=False,
        norm=False,
    ):
        super().__init__(lm_config, index_ds_size, clip_name, adpter, norm)
        self.freeze_prefix(freeze_prefix_list)

    def forward(self, img_input, ice_input, ice_seq_idx):
        inputs_embeds = super().forward(img_input, ice_seq_idx)
        if ice_input is None:
            lm_output = self.lm_model(inputs_embeds=inputs_embeds)
            return lm_output

        bs = len(img_input)
        ice_num = ice_input['pixel_values'].shape[0] // bs
        img_shape = ice_input['pixel_values'].shape[-3:]
        ice_input['pixel_values'] = ice_input['pixel_values'].view(-1, *img_shape)

        ice_img_features = self.img_model(ice_input['pixel_values'])['image_embeds']
        if self._adpter:
            ice_img_features = self.img_adpter(ice_img_features)
        if self._norm:
            ice_img_features = ice_img_features / ice_img_features.norm(
                dim=-1, keepdim=True
            )
        ice_img_features = ice_img_features.view(bs, ice_num, -1)

        inputs_embeds[:, 2 : 2 + ice_num] += ice_img_features

        output = self.lm_model(inputs_embeds=inputs_embeds, labels=ice_seq_idx)
        return output

    @torch.inference_mode()
    def generation(
        self,
        img_input,
        init_ice_idx,
        shot_num,
        index_ds,
        processor,
        device,
        image_field,
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

            ice_img_list = [
                index_ds[i][image_field] for i in ice_seq_idx.tolist()[0][2:]
            ]
            ice_input = processor(
                images=ice_img_list,
                padding=True,
                return_tensors='pt',
            ).to(device)
            # ice_input = {k: v.unsqueeze(dim=0) for k, v in ice_input.items()}

        return ice_seq_idx.detach().cpu().tolist()
