from io import BytesIO
from typing import List

import requests
import torch
from loguru import logger
from openicl import PromptTemplate
from PIL import Image

from .utils import cast_type, get_autocast, is_url
from .base_interface import BaseInterface


class LLMInterface(BaseInterface):
    def __init__(
        self,
        model,
        tokenizer,
        precision,
        device,
        input_ids_field_name,
        prompt_template,
        column_token_map,
        instruction,
        icd_join_char,
        label_field,
    ) -> None:
        super().__init__(
            precision=precision,
            device=device,
            input_ids_field_name=input_ids_field_name,
            prompt_template=prompt_template,
            column_token_map=column_token_map,
            instruction=instruction,
            icd_join_char=icd_join_char,
            label_field=label_field,
        )

        self.tokenizer = tokenizer
        self.model = model.to(self.device, self.data_type)

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_token_id = tokenizer.pad_token_id

    def concat_prompt(
        self,
        data_sample_list: list,
        add_eos_token: bool = False,
        is_last_for_generation: bool = True,
    ):
        """Return the concatenated prompt: <Instruction>text1<icd_join_char> ... textn[<icd_join_char>][</s>]

        Args:
            data_sample_list (List[DataSample]): List of data samples used to generate parts of the prompt.
            add_eos_token (bool, optional): Whether to add the EOS token at the end of the prompt. Defaults to False.
            is_last_for_generation (bool, optional): Whether the last data sample is used as a query for Generation inference. Defaults to True.

        Returns:
            str: Concatenated prompt string.
        """
        prompt = self.instruction
        if is_last_for_generation:
            query_prompt = self.gen_query_prompt(data_sample_list[-1])
            ice_sample_list = data_sample_list[:-1]
        else:
            ice_sample_list = data_sample_list
            query_prompt = ""

        ice_prompt_list = self.gen_ice_list_prompts(ice_sample_list)
        for ice_prompt in ice_prompt_list:
            prompt += ice_prompt.strip(" ") + self.icd_join_char

        prompt += query_prompt
        if is_last_for_generation:
            return prompt

        if add_eos_token:
            prompt += self.tokenizer.eos_token

        return prompt

    def prepare_input(
        self,
        batch_prompts,
        add_eos_token: bool = False,
        is_last_for_generation: bool = True,
        debug=False,
    ):
        if not any(isinstance(i, list) for i in batch_prompts):
            batch_prompts = [batch_prompts]
        prompts = []
        for sample in batch_prompts:
            prompt = self.concat_prompt(sample, add_eos_token, is_last_for_generation)
            if debug:
                print(f"{prompt=}")
            prompts.append(prompt)
        return self.tokenizer(prompts, return_tensors="pt", padding=True)
