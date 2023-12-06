from typing import Dict, List, Optional

import more_itertools
import numpy as np
import torch
from PIL import Image


def get_input_token_num(tokenizer, inputs: str):
    return len(tokenizer(inputs, verbose=False)['input_ids'])


@torch.inference_mode()
def get_ppl(
    model,
    model_input,
    autocast_context,
    icd_token_length=None,
    pad_token_id=0,
    left_padding_len=0,
):
    with autocast_context:
        outputs = model(**model_input)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = model_input["lang_x"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(
            reduction='none', ignore_index=pad_token_id
        )
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        loss = loss.view(shift_labels.size())

        if icd_token_length is not None:
            loss_mask = torch.zeros_like(shift_labels)  # [batch, seqlen]
            for i in range(len(loss_mask)):
                for j in range(icd_token_length[i] - 1, len(loss_mask[i])):
                    loss_mask[i][j] = 1
            loss = loss * loss_mask
        lens = (model_input["lang_x"] != pad_token_id).sum(-1)
        if icd_token_length is not None:
            lens -= torch.tensor(icd_token_length, device=lens.device)
        lens += left_padding_len
        ce_loss = loss.sum(-1) / lens
    return ce_loss


@torch.inference_mode()
def get_info_score(
    model,
    tokenizer,
    image_processor,
    device: str,
    icd_join_char: str,
    lang_x: List[str],
    image_x: List[Image.Image],
    candidate_set: Dict,
    batch_size: int,
    autocast_context,
    split_token: Optional[str] = None,
):
    model.eval()
    tokenizer.padding_side = "right"

    # 1. 计算P(y|x)
    # 1.1 拼接文本输入
    test_lang_x_input = lang_x[-1]
    chosen_icd_input = icd_join_char.join(lang_x[:-1])
    if chosen_icd_input:
        chosen_icd_input += icd_join_char

    query_test_lang_x_input = test_lang_x_input.split(split_token)[0] + split_token
    mask_context = chosen_icd_input + query_test_lang_x_input

    mask_length = get_input_token_num(tokenizer, mask_context)

    lang_x_input = chosen_icd_input + test_lang_x_input + icd_join_char
    lang_x_input = tokenizer(lang_x_input, return_tensors='pt').to(device=device)
    lang_x_input['attention_mask'][lang_x_input['input_ids'] == 0] = 0

    # 1.2 拼接图像输入
    image_x = [image_processor(image) for image in image_x]
    vision_x = torch.stack(image_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device=device, non_blocking=True)

    model_input = {
        'vision_x': vision_x,
        'lang_x': lang_x_input['input_ids'],
        'attention_mask': lang_x_input['attention_mask'].bool(),
    }
    ppl = get_ppl(
        model,
        model_input,
        autocast_context,
        icd_token_length=[mask_length],
        pad_token_id=tokenizer.pad_token_id,
    )

    # 2. 计算P(y|x, c)
    info_score_list = []
    cand_idx = sorted(list(candidate_set.keys()))
    for batch in more_itertools.chunked(cand_idx, batch_size):
        batch_data = [candidate_set[i] for i in batch]
        new_icd_lang_x = [data['text_input'] for data in batch_data]
        new_icd_image_x = [data['image'] for data in batch_data]

        # 2.1 拼接文本输入
        total_icd_lang_x_input = [
            icd_join_char.join([icd_lang_x] + lang_x) + icd_join_char
            for icd_lang_x in new_icd_lang_x
        ]
        total_icd_lang_x_input = tokenizer(
            total_icd_lang_x_input, return_tensors='pt', padding=True
        ).to(device=device)

        icd_text_list = [
            icd_join_char.join([icd_lang_x] + lang_x[:-1]) + icd_join_char
            for icd_lang_x in new_icd_lang_x
        ]

        total_icd_input_token_num = [
            get_input_token_num(tokenizer, icd_lang_x + query_test_lang_x_input)
            for icd_lang_x in icd_text_list
        ]

        # 2.2 拼接图像输入
        batch_total_vision_x = [
            torch.stack([image_processor(icd_image_x)] + image_x, dim=0)
            for icd_image_x in new_icd_image_x
        ]
        total_vision_x = torch.stack(batch_total_vision_x, dim=0)

        total_vision_x = total_vision_x.unsqueeze(2).to(
            device=device, non_blocking=True
        )
        model_input = {
            'vision_x': total_vision_x,
            'lang_x': total_icd_lang_x_input['input_ids'],
            'attention_mask': total_icd_lang_x_input['attention_mask'].bool(),
        }
        new_ppl = get_ppl(
            model,
            model_input,
            autocast_context,
            icd_token_length=total_icd_input_token_num,
            pad_token_id=tokenizer.pad_token_id,
        )
        sub_info_score = (-new_ppl).exp() - (-ppl).exp()
        info_score_list.append(sub_info_score)
    return torch.cat(info_score_list)
