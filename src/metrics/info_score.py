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
    ice_token_length=None,
    pad_token_id=0,
    left_padding_len=0,
):
    with autocast_context():
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

        if ice_token_length is not None:
            loss_mask = torch.zeros_like(shift_labels)  # [batch, seqlen]
            for i in range(len(loss_mask)):
                for j in range(ice_token_length[i] - 1, len(loss_mask[i])):
                    loss_mask[i][j] = 1
            loss = loss * loss_mask
        lens = (model_input["lang_x"] != pad_token_id).sum(-1)
        if ice_token_length is not None:
            lens -= torch.tensor(ice_token_length, device=lens.device)
        lens += left_padding_len
        ce_loss = loss.sum(-1) / lens
    return ce_loss


@torch.inference_mode()
def get_info_score(
    model,
    tokenizer,
    image_processor,
    device: str,
    ice_join_char: str,
    lang_x: List[str],
    image_x: List[Image.Image],
    candidate_set: Dict,
    batch_size: int,
    autocast_context,
    only_y_loss: bool = False,
    split_token: Optional[str] = None,
):
    model.eval()
    tokenizer.padding_side = "right"

    # 1. 计算P(y|x)
    # 1.1 拼接文本输入
    test_lang_x_input = lang_x[-1]
    chosen_ice_input = ice_join_char.join(lang_x[:-1])
    if chosen_ice_input:
        chosen_ice_input += '<|endofchunk|>'
    left_padding_token = 0
    if not chosen_ice_input and not only_y_loss:
        chosen_ice_input = '<|endoftext|>'
        left_padding_token = 1

    if only_y_loss:
        query_test_lang_x_input = test_lang_x_input.split(split_token)[0] + split_token
        mask_context = chosen_ice_input + query_test_lang_x_input
    else:
        mask_context = chosen_ice_input

    mask_length = get_input_token_num(tokenizer, mask_context)

    lang_x_input = chosen_ice_input + test_lang_x_input + ice_join_char
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
        ice_token_length=[mask_length],
        pad_token_id=tokenizer.pad_token_id,
        left_padding_len=left_padding_token,
    )

    # 2. 计算P(y|x, c)
    info_score_list = []
    cand_idx = sorted(list(candidate_set.keys()))
    for batch in more_itertools.chunked(cand_idx, batch_size):
        batch_data = [candidate_set[i] for i in batch]
        new_ice_lang_x = [data['text_input'] for data in batch_data]
        new_ice_image_x = [data['image'] for data in batch_data]

        # 2.1 拼接文本输入
        total_ice_lang_x_input = [
            ice_join_char.join([ice_lang_x] + lang_x) + ice_join_char
            for ice_lang_x in new_ice_lang_x
        ]
        total_ice_lang_x_input = tokenizer(
            total_ice_lang_x_input, return_tensors='pt', padding=True
        ).to(device=device)

        ice_text_list = [
            ice_join_char.join([ice_lang_x] + lang_x[:-1]) + ice_join_char
            for ice_lang_x in new_ice_lang_x
        ]
        if only_y_loss:
            total_ice_input_token_num = [
                get_input_token_num(tokenizer, ice_lang_x + query_test_lang_x_input)
                for ice_lang_x in ice_text_list
            ]
        else:
            total_ice_input_token_num = [
                get_input_token_num(tokenizer, ice_lang_x)
                for ice_lang_x in ice_text_list
            ]

        # 2.2 拼接图像输入
        batch_total_vision_x = [
            torch.stack([image_processor(ice_image_x)] + image_x, dim=0)
            for ice_image_x in new_ice_image_x
        ]
        total_vision_x = torch.stack(batch_total_vision_x, dim=0)

        total_vision_x = total_vision_x.unsqueeze(2).to(
            device=device, non_blocking=True
        )
        model_input = {
            'vision_x': total_vision_x,
            'lang_x': total_ice_lang_x_input['input_ids'],
            'attention_mask': total_ice_lang_x_input['attention_mask'].bool(),
        }
        new_ppl = get_ppl(
            model,
            model_input,
            autocast_context,
            ice_token_length=total_ice_input_token_num,
            pad_token_id=tokenizer.pad_token_id,
        )
        sub_info_score = (-new_ppl).exp() - (-ppl).exp()
        info_score_list.append(sub_info_score)
    return torch.cat(info_score_list)
