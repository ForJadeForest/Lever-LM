from io import BytesIO
from typing import List

import requests
import torch
from loguru import logger
from openicl import PromptTemplate
from PIL import Image

from .utils import cast_type, get_autocast, is_url


class BaseInterface(torch.nn.Module):
    def __init__(
        self,
        precision,
        device,
        input_ids_field_name,
        prompt_template,
        column_token_map,
        instruction,
        icd_join_char,
        label_field,
        image_field,
    ) -> None:
        super().__init__()
        self.data_type = cast_type(precision)
        self.autocast_context = get_autocast(precision)
        self.device = device
        self.input_ids_field_name = input_ids_field_name

        self.pt = PromptTemplate(
            prompt_template,
            column_token_map=dict(column_token_map),
        )
        self.icd_join_char = icd_join_char
        self.instruction = instruction
        self.pad_token_id = None
        self.tokenizer = None
        self.label_field = label_field
        self.image_field = image_field

    def gen_query_prompt(self, query, add_image_token=True) -> str:
        """Generate a query prompt without a label. Includes an image token if specified.

        For a Caption example with add_image_token=True, the format would be:
        [<IMAGE_TOKEN>]Caption:

        Args:
            query (DataSample): Query data sample.
            add_image_token (bool, optional): Whether to add an image token. Defaults to True.

        Returns:
            str: Query sample prompt.
        """
        query_prompt = self.pt.generate_item(query, output_field=self.label_field)
        if add_image_token:
            return self.add_image_token(query_prompt)
        return query_prompt

    def gen_ice_prompt(self, ice, add_image_token=True) -> str:
        """Generate an In-context Example (ICE) prompt with a label. Includes an image token if specified.

        For a Caption example with add_image_token=True, the format would be:
        <IMAGE_TOKEN>Caption: This is a cat.

        Args:
            ice (DataSample): ICE data sample.
            add_image_token (bool, optional): Whether to add an image token. Defaults to True.

        Returns:
            str: In-context Example sample prompt.
        """
        ice_prompt = self.pt.generate_item(ice)
        if add_image_token:
            return self.add_image_token(ice_prompt)
        return ice_prompt

    def gen_ice_list_prompts(self, ice_list: list, add_image_token=True) -> List[str]:
        """Generate a list of In-context Example (ICE) prompts from a list of ICE data samples.

        Args:
            ice_list (List[YourDataType]): List of ICE data samples.
            add_image_token (bool, optional): Whether to add an image token to each prompt. Defaults to True.

        Returns:
            List[str]: List of ICE sample prompts.
        """
        return [self.gen_ice_prompt(ice, add_image_token) for ice in ice_list]

    def concat_prompt(self, *args, **kwargs):
        raise NotImplemented

    def prepare_input(self, *args, **kwargs):
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
            shift_labels = model_input[self.input_ids_field_name][..., 1:].contiguous()
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
            lens = (model_input[self.input_ids_field_name] != self.pad_token_id).sum(-1)

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

    def transfer_prompts(self, batch_data_sample_list, is_last_for_generation=True):
        if not any(isinstance(i, list) for i in batch_data_sample_list):
            batch_data_sample_list = [batch_data_sample_list]

        prompts = []
        for data_sample_list in batch_data_sample_list:
            prompt = []
            for data_sample in data_sample_list[:-1]:
                prompt.extend(
                    [
                        data_sample[self.image_field],
                        self.gen_ice_prompt(data_sample, add_image_token=False),
                    ]
                )
            prompt.append(data_sample_list[-1][self.image_field])
            if is_last_for_generation:
                prompt.append(
                    self.gen_query_prompt(data_sample_list[-1], add_image_token=False)
                )
            else:
                prompt.append(
                    self.gen_ice_prompt(data_sample_list[-1], add_image_token=False)
                )

            prompts.append(prompt)
        return prompts

    def transfer_prompts_text_part(
        self, batch_data_sample_list, is_last_for_generation=True
    ):
        if not any(isinstance(i, list) for i in batch_data_sample_list):
            batch_data_sample_list = [batch_data_sample_list]

        prompts = []
        for data_sample_list in batch_data_sample_list:
            prompt = []
            for data_sample in data_sample_list[:-1]:
                prompt.append(
                    self.gen_ice_prompt(data_sample, add_image_token=False),
                )
            if is_last_for_generation:
                prompt.append(
                    self.gen_query_prompt(data_sample_list[-1], add_image_token=False)
                )
            else:
                prompt.append(
                    self.gen_ice_prompt(data_sample_list[-1], add_image_token=False)
                )

            prompts.append(prompt)
        return prompts

    def generate(self, *args, **kwargs):
        with self.autocast_context:
            return self.model.generate(*args, **kwargs)

    def __call__(self, model_input):
        with self.autocast_context:
            return self.model(**model_input)
