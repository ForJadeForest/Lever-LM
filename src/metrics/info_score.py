from typing import Dict, List

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
        lens = (model_input["lang_x"] != pad_token_id).sum(-1).cpu().numpy()
        if ice_token_length is not None:
            lens -= np.array(ice_token_length)
        lens += left_padding_len
        ce_loss = loss.sum(-1).cpu().detach() / lens
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
):
    model.eval()
    tokenizer.padding_side = "right"

    # 1. 计算P(y|x)
    # 1.1 拼接文本输入
    test_lang_x_input = lang_x[-1]
    chosen_ice_input = ice_join_char.join(lang_x[:-1]) + ice_join_char
    if not chosen_ice_input:
        chosen_ice_input = '<|endoftext|>'
    chosen_ice_len = get_input_token_num(tokenizer, chosen_ice_input)

    lang_x_input = chosen_ice_input + test_lang_x_input
    lang_x_input = tokenizer(lang_x_input, return_tensors='pt').to(device=device)

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
        ice_token_length=[chosen_ice_len],
        pad_token_id=tokenizer.pad_token_id,
        left_padding_len=1,
    )

    # 2. 计算P(y|x, c)
    info_score_list = []
    cand_idx = sorted(list(candidate_set.keys()))
    for batch in more_itertools.chunked(cand_idx, batch_size):
        batch_data = [candidate_set[i] for i in batch]
        new_ice_lang_x = [data['caption'] for data in batch_data]
        new_ice_image_x = [
            Image.open(data['image']).convert('RGB') for data in batch_data
        ]

        # 2.1 拼接文本输入
        total_ice_lang_x_input = [
            ice_join_char.join([ice_lang_x] + lang_x) for ice_lang_x in new_ice_lang_x
        ]
        total_ice_lang_x_input = tokenizer(
            total_ice_lang_x_input, return_tensors='pt', padding=True
        ).to(device=device)

        ice_text_list = [
            ice_join_char.join([ice_lang_x] + lang_x[:-1]) + ice_join_char
            for ice_lang_x in new_ice_lang_x
        ]
        total_ice_input_token_num = [
            get_input_token_num(tokenizer, ice_lang_x) for ice_lang_x in ice_text_list
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
        info_score_list.extend(sub_info_score.cpu().numpy().tolist())
    return info_score_list


# @torch.inference_mode()
# def get_info_score(
#     model,
#     tokenizer,
#     image_processor,
#     device,
#     ice_join_char: str,
#     lang_x: List[str],
#     new_ice_lang_x: List[str],
#     image_x: List[Image.Image],
#     new_ice_image_x: List[Image.Image],
#     autocast_context,
# ):
#     """
#     lang_x: 已经选择好的ice + 测试样本(lang_x[-1])
#     new_ice_lang_x: batch个待计算分数的样本
#     """
#     model.eval()
#     tokenizer.padding_side = "right"

#     # 1. 计算P(y|x)
#     # 1.1 拼接文本输入
#     test_lang_x_input = lang_x[-1]
#     chosen_ice_input = ice_join_char.join(lang_x[:-1]) + ice_join_char
#     chosen_ice_len = get_input_token_num(tokenizer, chosen_ice_input)

#     lang_x_input = chosen_ice_input + test_lang_x_input
#     lang_x_input = tokenizer(lang_x_input, return_tensors='pt').to(device=device)

#     # 2.2 拼接图像输入
#     image_x = [image_processor(image).unsqueeze(0) for image in image_x]
#     vision_x = torch.cat(image_x, dim=0)
#     vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(device=device, non_blocking=True)

#     model_input = {
#         'vision_x': vision_x,
#         'lang_x': lang_x_input['input_ids'],
#         'attention_mask': lang_x_input['attention_mask'].bool(),
#     }
#     ppl = get_ppl(
#         model,
#         model_input,
#         autocast_context,
#         ice_token_length=[chosen_ice_len],
#         pad_token_id=tokenizer.pad_token_id,
#     )

#     # 2. 计算P(y|x, c)
#     # 2.1 拼接文本输入
#     total_ice_lang_x_input = [
#         ice_join_char.join([ice_lang_x] + lang_x) for ice_lang_x in new_ice_lang_x
#     ]
#     total_ice_lang_x_input = tokenizer(
#         total_ice_lang_x_input, return_tensors='pt', padding=True
#     ).to(device=device)
#     ice_text_list = [
#         ice_join_char.join([ice_lang_x] + lang_x[:-1]) + ice_join_char
#         for ice_lang_x in new_ice_lang_x
#     ]
#     total_ice_input_token_num = [
#         get_input_token_num(tokenizer, ice_lang_x) for ice_lang_x in ice_text_list
#     ]
#     # 2.2 拼接图像输入
#     batch_total_vision_x = [
#         torch.cat([image_processor(ice_image_x).unsqueeze(0)] + image_x, dim=0)
#         for ice_image_x in new_ice_image_x
#     ]
#     total_vision_x = torch.stack(batch_total_vision_x, dim=0)

#     total_vision_x = total_vision_x.unsqueeze(2).to(device=device, non_blocking=True)

#     model_input = {
#         'vision_x': total_vision_x,
#         'lang_x': total_ice_lang_x_input['input_ids'],
#         'attention_mask': total_ice_lang_x_input['attention_mask'].bool(),
#     }
#     new_ppl = get_ppl(
#         model,
#         model_input,
#         autocast_context,
#         ice_token_length=total_ice_input_token_num,
#         pad_token_id=tokenizer.pad_token_id,
#     )

#     info_score = (-new_ppl).exp() - (-ppl).exp()

#     return info_score
