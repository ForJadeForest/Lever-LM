from typing import Dict, List, Optional, Union

import more_itertools
import torch
from PIL import Image

from src.models.lvlms import FlamingoInterface, IDEFICSInterface


@torch.inference_mode()
def get_info_score(
    interface: Union[FlamingoInterface, IDEFICSInterface],
    choosed_icd_seq_list: List,
    candidate_set: Dict,
    batch_size: int,
    split_token: Optional[str] = None,
    construct_order='left',
):
    # 1. 计算P(y|x)
    # 1.1 拼接文本输入
    test_lang_x_input = interface.gen_ice_prompt(
        choosed_icd_seq_list[-1], add_image_token=True
    )
    prompts = interface.transfer_prompts(
        choosed_icd_seq_list, is_last_for_generation=False
    )

    x_input = interface.prepare_input(
        prompts, is_last_for_generation=False, add_eos_token=True
    ).to(interface.device)

    icd_mask_prompt = interface.concat_prompt(
        choosed_icd_seq_list[:-1],
        add_eos_token=False,
        add_image_token=True,
        is_last_for_generation=False,
    )
    query_mask_part = test_lang_x_input.split(split_token)[0] + split_token

    mask_context = icd_mask_prompt + query_mask_part

    mask_length = interface.get_input_token_num(mask_context)
    cond_prob = interface.get_cond_prob(x_input, mask_length=[mask_length])

    # 2. 计算P(y|x, c)
    info_score_list = []
    cand_idx = sorted(list(candidate_set.keys()))
    for batch in more_itertools.chunked(cand_idx, batch_size):
        batch_data = [candidate_set[i] for i in batch]

        # 2.1 拼接文本输入
        if construct_order == 'left':
            add_new_icd_seq_list = [
                [new_icd] + choosed_icd_seq_list for new_icd in batch_data
            ]
        elif construct_order == 'right':
            add_new_icd_seq_list = [
                choosed_icd_seq_list[:-1] + [new_icd] + [choosed_icd_seq_list[-1]]
                for new_icd in batch_data
            ]
        else:
            raise ValueError(
                f"the construct_order should be left or right, but got {construct_order}"
            )

        prompts = interface.transfer_prompts(
            add_new_icd_seq_list, is_last_for_generation=False
        )

        add_new_icd_input = interface.prepare_input(
            prompts,
            is_last_for_generation=False,
            add_eos_token=True,
        ).to(interface.device)
        icd_mask_prompt_list = [
            interface.concat_prompt(
                t[:-1],
                add_eos_token=False,
                add_image_token=True,
                is_last_for_generation=False,
            )
            for t in add_new_icd_seq_list
        ]

        mask_context_list = [
            icd_mask_prompt + query_mask_part
            for icd_mask_prompt in icd_mask_prompt_list
        ]

        mask_length_list = [
            interface.get_input_token_num(mask_context)
            for mask_context in mask_context_list
        ]
        # for i in range(32):
        #     print(interface.tokenizer.decode(add_new_icd_input['lang_x'][i, mask_length_list[i]:]))

        new_cond_prob = interface.get_cond_prob(
            add_new_icd_input, mask_length=mask_length_list
        )
        sub_info_score = new_cond_prob - cond_prob
        info_score_list.append(sub_info_score)
    return torch.cat(info_score_list)
