from io import BytesIO
from typing import List, Optional

from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_interface import BaseInterface


class LLMInterface(BaseInterface):
    def __init__(
        self,
        model_or_model_name,
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
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

        self.model = model_or_model_name
        if isinstance(self.model, str):
            self.model = AutoModelForCausalLM.from_pretrained(model_or_model_name)
        self.model = self.model.to(self.device, self.data_type)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

    def transfer_prompts(
        self, batch_data_sample_list, is_last_for_generation=True, query_label=None
    ):
        if not any(isinstance(i, list) for i in batch_data_sample_list):
            batch_data_sample_list = [batch_data_sample_list]

        prompts = []
        for data_sample_list in batch_data_sample_list:
            prompt = []
            for data_sample in data_sample_list[:-1]:
                prompt.append(
                    self.gen_text_with_label(data_sample),
                )
            if is_last_for_generation:
                prompt.append(self.gen_text_without_label(data_sample_list[-1]))
            else:
                prompt.append(
                    self.gen_text_with_label(data_sample_list[-1], label=query_label)
                )

            prompts.append(prompt)
        return prompts

    def concat_prompt(
        self,
        data_sample_list: list,
        add_eos_token: bool = False,
        is_last_for_generation: bool = True,
        query_label: Optional[int] = None,
    ):
        """Return the concatenated prompt: <Instruction>text1<icd_join_char> ... textn[<icd_join_char>][</s>]

        Args:
            data_sample_list (List[DataSample]): List of data samples used to generate parts of the prompt.
            add_eos_token (bool, optional): Whether to add the EOS token at the end of the prompt. Defaults to False.
            is_last_for_generation (bool, optional): Whether the last data sample is used as a query for Generation inference. Defaults to True.

        Returns:
            str: Concatenated prompt string.
        """
        if len(data_sample_list) == 0:
            return ""
        prompt = self.instruction
        ice_data_sample_list = data_sample_list[:-1]
        query_data_sample = data_sample_list[-1]
        if is_last_for_generation:
            query_prompt = self.gen_text_without_label(query_data_sample)
        else:
            query_prompt = self.gen_text_with_label(query_data_sample, query_label)

        ice_prompt_list = [
            self.gen_text_with_label(item) for item in ice_data_sample_list
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
        add_eos_token: bool = False,
        is_last_for_generation: bool = True,
        debug=False,
    ):
        if not any(isinstance(i, list) for i in batch_prompts):
            batch_prompts = [batch_prompts]
        prompts = []
        for sample_list in batch_prompts:
            prompt = ""
            for i, p in enumerate(sample_list):
                prompt += p.strip(" ")
                if i != len(sample_list) - 1 or not is_last_for_generation:
                    prompt += self.icd_join_char
            if add_eos_token and not is_last_for_generation:
                prompt += self.tokenizer.eos_token

            if debug:
                print(f"{prompt=}")
            prompts.append(prompt)
        return self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )
