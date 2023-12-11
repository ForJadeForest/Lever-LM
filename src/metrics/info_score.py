from typing import Dict, List, Optional, Union

import more_itertools
import torch
from PIL import Image

from src.models.lvlms import FlamingoInterface, IDEFICSInterface


@torch.inference_mode()
def get_info_score(
    interface: Union[FlamingoInterface, IDEFICSInterface],
    text_input_list: List[str],
    image_input_list: List[Image.Image],
    candidate_set: Dict,
    batch_size: int,
    split_token: Optional[str] = None,
    construct_order='left',
):
    # 1. 计算P(y|x)
    # 1.1 拼接文本输入
    test_lang_x_input = text_input_list[-1]
    prompts = interface.transfer_prompts(text_input_list, image_input_list)

    x_input = interface.prepare_input(
        prompts, add_join_token_end=True, add_eos_token=True
    )

    icd_mask_prompt = interface.construct_prompt(
        text_input_list[:-1],
        add_join_token_end=True,
        add_eos_token=False,
        add_image_token=True,
    )
    query_mask_part = interface.add_image_token(
        test_lang_x_input.split(split_token)[0] + split_token
    )
    mask_context = icd_mask_prompt + query_mask_part

    mask_length = interface.get_input_token_num(mask_context)
    cond_prob = interface.get_cond_prob(x_input, mask_length=[mask_length])

    # 2. 计算P(y|x, c)
    info_score_list = []
    cand_idx = sorted(list(candidate_set.keys()))
    for batch in more_itertools.chunked(cand_idx, batch_size):
        batch_data = [candidate_set[i] for i in batch]
        new_icd_text_list = [data['text_input'] for data in batch_data]
        new_icd_image_list = [data['image'] for data in batch_data]

        # 2.1 拼接文本输入
        if construct_order == 'left':
            add_new_icd_texts = [
                [new_icd_text] + text_input_list for new_icd_text in new_icd_text_list
            ]
            add_new_icd_images = [
                [new_icd_image] + image_input_list
                for new_icd_image in new_icd_image_list
            ]
        elif construct_order == 'right':
            add_new_icd_texts = [
                text_input_list[:-1] + [new_icd_text] + [text_input_list[-1]]
                for new_icd_text in new_icd_text_list
            ]
            add_new_icd_images = [
                image_input_list[:-1] + [new_icd_image] + [image_input_list[-1]]
                for new_icd_image in new_icd_image_list
            ]
        else:
            raise ValueError(
                f"the construct_order should be left or right, but got {construct_order}"
            )

        prompts = interface.transfer_prompts(add_new_icd_texts, add_new_icd_images)

        add_new_icd_input = interface.prepare_input(
            prompts,
            add_join_token_end=True,
            add_eos_token=True,
        )
        icd_mask_prompt_list = [
            interface.construct_prompt(
                t[:-1],
                add_join_token_end=True,
                add_eos_token=False,
                add_image_token=True,
            )
            for t in add_new_icd_texts
        ]

        mask_context_list = [
            icd_mask_prompt + query_mask_part
            for icd_mask_prompt in icd_mask_prompt_list
        ]

        mask_length_list = [
            interface.get_input_token_num(mask_context)
            for mask_context in mask_context_list
        ]

        # interface.tokenizer.decode(add_new_icd_input['input_ids'][:, mask_length_list:])

        new_cond_prob = interface.get_cond_prob(
            add_new_icd_input, mask_length=mask_length_list
        )
        sub_info_score = new_cond_prob - cond_prob
        info_score_list.append(sub_info_score)
    return torch.cat(info_score_list)
