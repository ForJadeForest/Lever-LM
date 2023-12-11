from io import BytesIO

import requests
import torch
from loguru import logger
from openicl import PromptTemplate
from PIL import Image

from .utils import cast_type, get_autocast, is_url


class BaseInterface:
    def __init__(
        self,
        precision,
        device,
        input_ids_filed_name,
        prompt_template,
        column_token_map,
        icd_token,
        instruction,
        icd_join_char,
    ) -> None:
        self.data_type = cast_type(precision)
        self.autocast_context = get_autocast(precision)
        self.device = device
        self.input_ids_filed_name = input_ids_filed_name

        self.pt = PromptTemplate(
            prompt_template,
            column_token_map=dict(column_token_map),
            ice_token=icd_token,
        )
        self.icd_join_char = icd_join_char
        self.instruction = instruction
        self.pad_token_id = None
        self.tokenizer = None

    def construct_icd_prompt(self, data):
        return self.pt.generate_item(data)

    def construct_prompt(self, *args, **kwargs):
        raise NotImplemented

    def prepare_input(self, batch_text_list, batch_image_list, *args, **kwargs):
        raise NotImplemented

    def add_image_token(self, text):
        raise NotImplemented
    
    @torch.inference_mode()
    def get_cond_prob(
        self,
        model_input,
        mask_length=None,
    ):
        if self.pad_token_id is None:
            logger.warning('the pad_token_id is None')
        with self.autocast_context:
            outputs = self.model(**model_input)

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = model_input[self.input_ids_filed_name][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction='none', ignore_index=self.pad_token_id
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            loss = loss.view(shift_labels.size())

            if mask_length is not None:
                loss_mask = torch.zeros_like(shift_labels)  # [batch, seqlen]
                for i in range(len(loss_mask)):
                    for j in range(mask_length[i] - 1, len(loss_mask[i])):
                        loss_mask[i][j] = 1
                loss = loss * loss_mask
            lens = (model_input[self.input_ids_filed_name] != self.pad_token_id).sum(-1)

            if mask_length is not None:
                lens -= torch.tensor(mask_length, device=lens.device)
            # logger.debug(f'{lens=}')
            ce_loss = loss.sum(-1) / lens
        return (-ce_loss).exp()

    def get_input_token_num(self, input_tokens: str) -> int:
        return len(self.tokenizer(input_tokens, add_special_tokens=False)['input_ids'])

    def is_img(self, obj):
        if isinstance(obj, Image.Image):
            return obj
        elif isinstance(obj, str):
            if is_url(obj):
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0"
                        " Safari/537.36"
                    )
                }
                response = requests.get(obj, stream=True, headers=headers)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            else:
                try:
                    return Image.open(obj)
                except:
                    return None

    def transfer_prompts(self, batch_text_list, batch_image_list):
        if not any(isinstance(i, list) for i in batch_text_list):
            batch_text_list = [batch_text_list]
            batch_image_list = [batch_image_list]

        if len(batch_text_list) != len(batch_image_list):
            raise ValueError(
                f'the batch size of text_list should equal to the batch size of image_list, but {len(batch_text_list)=} != {len(batch_image_list)=}'
            )
        prompts = []
        for text_sample, image_sample in zip(batch_text_list, batch_image_list):
            if len(text_sample) != len(image_sample):
                raise ValueError(
                    f'the length of text_list should equal to the length of image_list, but {len(text_sample)=} != {len(image_sample)=}'
                )
            prompt = []
            for t, i in zip(text_sample, image_sample):
                prompt.extend([i, t])
            prompts.append(prompt)
        return prompts
