import os

import faiss
import more_itertools
import torch
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from src.models.lvlms import FlamingoInterface, IDEFICSInterface


def init_lvlm(cfg, **kwargs) -> FlamingoInterface:
    if 'flamingo' in cfg.lvlm.name:
        return FlamingoInterface(
            lang_encoder_path=cfg.lvlm.lang_encoder_path,
            tokenizer_path=cfg.lvlm.tokenizer_path,
            flamingo_checkpoint_dir=cfg.lvlm.flamingo_checkpoint_dir,
            cross_attn_every_n_layers=cfg.lvlm.cross_attn_every_n_layers,
            hf_root=cfg.lvlm.hf_root,
            precision=cfg.precision,
            device=kwargs['device'],
            prompt_template=cfg.task.template,
            column_token_map=cfg.task.column_token_map,
            icd_join_char=cfg.lvlm.icd_join_char,
            icd_token=cfg.task.icd_token,
            load_from_local=cfg.lvlm.load_from_local,
            instruction=cfg.task.instruction,
            init_device=cfg.lvlm.init_device,
        )
    elif 'idefics' in cfg.lvlm.name:
        return IDEFICSInterface(
            hf_root=cfg.lvlm.hf_root,
            load_from_local=cfg.lvlm.load_from_local,
            precision=cfg.precision,
            device=kwargs['device'],
            prompt_template=cfg.task.template,
            column_token_map=cfg.task.column_token_map,
            icd_token=cfg.task.icd_token,
            instruction=cfg.task.instruction,
            icd_join_char=cfg.lvlm.icd_join_char,
        )
    else:
        raise ValueError('LVLM name error, now only support [\'flamingo, idefics\']')


def recall_sim_feature(test_vec, train_vec, top_k=200):
    logger.info(f'embedding shape: {train_vec.shape}')
    dim = train_vec.shape[-1]
    index_feat = faiss.IndexFlatIP(dim)
    index_feat.add(train_vec)
    dist, index = index_feat.search(test_vec, top_k)
    return dist, index


@torch.inference_mode()
def encode_text(
    train_ds,
    data_key,
    device,
    model_type='openai/clip-vit-large-patch14',
    batch_size=128,
):
    model = CLIPTextModelWithProjection.from_pretrained(model_type).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    final_text_feature = []

    for batch in more_itertools.chunked(tqdm(train_ds), batch_size):
        text_list = [i[data_key] for i in batch]
        inputs = tokenizer(text_list, padding=True, return_tensors="pt").to(device)
        text_feature = model(**inputs).text_embeds
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        final_text_feature.append(text_feature)

    final_text_feature = torch.cat(final_text_feature, dim=0)
    return final_text_feature.detach().cpu().numpy()


@torch.inference_mode()
def encode_image(
    train_ds,
    data_key,
    device,
    model_type='openai/clip-vit-large-patch14',
    batch_size=128,
):
    model = CLIPVisionModelWithProjection.from_pretrained(model_type).to(device)
    processor = AutoProcessor.from_pretrained(model_type)
    model.eval()

    final_image_feature = []
    for batch in more_itertools.chunked(tqdm(train_ds), batch_size):
        images = [i[data_key] for i in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        image_feature = model(**inputs).image_embeds
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        final_image_feature.append(image_feature)

    final_image_feature = torch.cat(final_image_feature, dim=0)
    return final_image_feature.detach().cpu().numpy()


def data_split(generated_data, train_ratio):
    # 获得有多少条test数据
    test_dataset_id_set = {
        v[-1] for d in generated_data for v in generated_data[d]['id_list']
    }
    test_dataset_len = len(test_dataset_id_set)

    # 计算多少test数据用于训练 剩下部分用于监督val loss
    train_data_len = int(train_ratio * test_dataset_len)
    train_idx_set = set(sorted(list(test_dataset_id_set))[:train_data_len])
    val_idx_set = test_dataset_id_set - train_idx_set

    train_data_list = list()
    val_data_list = list()
    for d in generated_data:
        for i in generated_data[d]['id_list']:
            if int(i[-1]) in train_idx_set:
                train_data_list.append(i)
            elif int(i[-1]) in val_idx_set:
                val_data_list.append(i)
            else:
                raise ValueError()

    print(f'the train size {len(train_data_list)}, the test size {len(val_data_list)}')
    return train_data_list, val_data_list


def collate_fn(batch, processor: CLIPProcessor):
    bs = len(batch)
    collate_dict = {
        'icd_seq_idx': torch.stack([item['icd_seq_idx'] for item in batch]),
    }
    query_input = [d['query_input'] for d in batch]

    query_text_input = (
        [q['text'] for q in query_input] if 'text' in query_input[0] else None
    )
    query_image_input = (
        [q['images'] for q in query_input] if 'images' in query_input[0] else None
    )
    if query_text_input or query_image_input:
        query_input = processor(
            images=query_image_input,
            text=query_text_input,
            padding=True,
            return_tensors='pt',
        )
        collate_dict['query_input'] = query_input

    icd_input_list = [d['icd_input'] for d in batch]
    icd_image_input = icd_text_input = None
    if 'text' in icd_input_list[0]:
        icd_num = len(icd_input_list[0]['text'])
        icd_text_input = [i['text'] for i in icd_input_list]
        icd_text_input = [i for icd_text in icd_text_input for i in icd_text]
    if 'images' in icd_input_list[0]:
        icd_num = len(icd_input_list[0]['images'])
        icd_image_input = [i['images'] for i in icd_input_list]
        icd_image_input = [i for icd_image in icd_image_input for i in icd_image]

    if icd_image_input or icd_text_input:
        icd_input = processor(
            images=icd_image_input,
            text=icd_text_input,
            padding=True,
            return_tensors='pt',
        )
        if 'input_ids' in icd_input:
            icd_input['input_ids'] = icd_input['input_ids'].view(bs, icd_num, -1)
            icd_input['attention_mask'] = icd_input['attention_mask'].view(
                bs, icd_num, -1
            )
        if 'pixel_values' in icd_input:
            icd_input['pixel_values'] = icd_input['pixel_values'].view(
                bs, icd_num, *icd_input['pixel_values'].shape[1:]
            )
        collate_dict['icd_input'] = icd_input
    return collate_dict


def load_feature_cache(cfg, cache_path, encoding_method, train_ds, data_key):
    if os.path.exists(cache_path):
        features = torch.load(cache_path)
    else:
        features = encoding_method(
            train_ds,
            data_key,
            cfg.device,
            cfg.sim_model_type,
            cfg.candidate_set_encode_bs,
        )
        torch.save(features, cache_path)
    return features


def beam_filter(score_list, data_id_list, beam_size):
    score_list = torch.tensor(score_list)
    score_value, indices = torch.topk(score_list, beam_size)
    return score_value.tolist(), [data_id_list[idx] for idx in indices]
