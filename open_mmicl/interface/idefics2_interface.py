from typing import Optional

import torch
from loguru import logger

from .base_interface import LVLMInterface


class IDEFICS2Interface(LVLMInterface):
    def __init__(
        self,
        model_name_or_path,
        precision,
        device,
        prompt_template,
        column_token_map,
        instruction,
        image_field,
        label_field,
        icd_join_char="\n",
    ):
        from transformers import Idefics2ForConditionalGeneration, Idefics2Processor

        super().__init__(
            precision=precision,
            device=device,
            input_ids_field_name="input_ids",
            prompt_template=prompt_template,
            column_token_map=column_token_map,
            instruction=instruction,
            icd_join_char=icd_join_char,
            image_field=image_field,
            label_field=label_field,
        )
        self.processor = Idefics2Processor.from_pretrained(model_name_or_path)
        self.model = Idefics2ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=self.data_type,
        ).to(self.device)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.image_processor = self.processor.image_processor
        self.pad_token_id = self.tokenizer.pad_token_id

        self.image_token = self.processor.image_token.content
        self.fake_image_token = self.processor.fake_image_token.content
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

    def add_image_token(self, text):
        return (
            self.fake_image_token
            + self.processor.image_seq_len * self.image_token
            + self.fake_image_token
            + text
        )

    def concat_prompt(
        self,
        data_sample_list: list,
        add_eos_token: bool = False,
        add_image_token: bool = True,
        is_last_for_generation: bool = True,
        query_label: Optional[int] = None,
    ):
        """Return the concatenated prompt: <Instruction>[<IMAGE_TOKEN>]text1<icd_join_char> ... textn[<icd_join_char>][</s>]
        Note: Only support one image and one text pair.
        Args:
            data_sample_list (List[DataSample]): List of data samples used to generate parts of the prompt.
            add_eos_token (bool, optional): Whether to add the EOS token at the end of the prompt. Defaults to False.
            add_image_token (bool, optional): Whether to add an image token for each sample. Defaults to True.
            is_last_for_infer (bool, optional): Whether the last data sample is used as a query for Generation inference. Defaults to True.

        Returns:
            str: Concatenated prompt string.
        """
        prompt = self.tokenizer.bos_token + self.instruction
        if len(data_sample_list) == 0:
            return ""
        ice_data_sample_list = data_sample_list[:-1]
        query_data_sample = data_sample_list[-1]

        if is_last_for_generation:
            query_prompt = self.gen_text_without_label(
                query_data_sample, add_image_token=add_image_token
            )
        else:
            query_prompt = self.gen_text_with_label(
                query_data_sample, query_label, add_image_token
            )

        ice_prompt_list = [
            self.gen_text_with_label(item, add_image_token=add_image_token)
            for item in ice_data_sample_list
        ]
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
        add_eos_token=False,
        add_special_tokens=True,
        padding="longest",
        truncation=None,
        max_length=None,
        return_tensors="pt",
        is_last_for_generation=False,
    ):
        batch_text_inputs = []
        batch_image_inputs = []
        for prompts in batch_prompts:
            image_inputs = []
            text_inputs = ""
            for item in prompts:
                if self.is_img(item):
                    image_inputs.append(item)
                    text_inputs += "<image>"
                else:
                    text_inputs += item
            if add_eos_token:
                text_inputs += "</s>"
            batch_text_inputs.append(text_inputs)
            batch_image_inputs.append(image_inputs)

        return self.processor(
            text=batch_text_inputs,
            images=batch_image_inputs,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
