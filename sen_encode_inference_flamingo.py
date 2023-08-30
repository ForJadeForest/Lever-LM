import datetime
import gc
import json
import os

import pandas as pd
import torch
from openicl import (
    DatasetReader,
    DirRetriever,
    FlamingoGenInferencer,
    ICLMRetriever,
    PromptTemplate,
    RandomRetriever,
    TopkRetriever,
    ZeroRetriever,
)
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import BertTokenizer, GPT2Config

from datasets import load_dataset
from src.datasets import CocoDataset, SentenceEncodeCaptionICLMDataset
from src.models import SenEncodeCaptionICLM
from src.utils import init_flamingo


def compute_cider(
    result_dict,
    annotations_path,
):
    # create coco object and coco_result object
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_dict)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    return coco_eval.eval


def load_coco_dict(root_path, annotations_file):
    dataset = CocoDataset(root_path, annotations_file)

    image_id_list = []
    single_caption_list = []
    captions_list = []
    file_name_list = []
    for d in dataset:
        image_id_list.append(d['image_id'])
        single_caption_list.append(d['single_caption'])
        file_name_list.append(os.path.basename(d['image']))
        captions_list.append(d['captions'])

    data_dict = {
        'image_id': image_id_list,
        'single_caption': single_caption_list,
        'captions': captions_list,
        'file_name': file_name_list,
    }
    return data_dict


def get_cider(inferencer, retriever, ice_prompt, num_shot):
    output_dict = inferencer.inference(
        retriever,
        ice_prompt,
        output_json_filename=f'{str(datetime.datetime.now())}-{type(inferencer).__name__}-{type(retriever).__name__}-coco2017val_flamingo-{num_shot=}-{test_data_num=}',
        return_dict=True,
    )
    pred_coco = []
    for idx in output_dict:
        pred_coco.append(
            {
                'image_id': output_dict[idx]['image_id'],
                'caption': output_dict[idx]['prediction']
                .split("Output", 1)[0]
                .replace('"', ""),
            }
        )

    ann_path = os.path.join(data_root_dir, 'annotations', 'captions_val2017.json')
    cider_score = compute_cider(pred_coco, ann_path)['CIDEr']
    return cider_score


if __name__ == '__main__':
    overwrite = False
    test_random = True
    teat_zero_shot = True
    test_topk_caption = True
    test_iclm = True

    test_data_num = 500
    result_json_path = './res.json'
    data_root_dir = '/data/share/pyz/data/mscoco/mscoco2017/'
    iclm_model_path = '/home/pyz32/code/iclm/result/model_cpk/sentence_encode_iclm/epoch:8-min_loss:11.3398.pth'
    test_emb_path = '/home/pyz32/code/ICLM/open_flamingo_caption/result/cache/coco_val_flamingo_mean_feature.pth'

    lang_encoder_path = 'anas-awadalla/mpt-7b'
    tokenizer_path = 'anas-awadalla/mpt-7b'
    flamingo_checkpoint_path = (
        '/data/share/pyz/checkpoint/openflamingo/OpenFlamingo-9B-vitl-mpt7b'
    )
    cross_attn_every_n_layers = 4
    hf_root = 'OpenFlamingo-9B-vitl-mpt7b'
    # lang_encoder_path = 'anas-awadalla/mpt-1b-redpajama-200b'
    # tokenizer_path = 'anas-awadalla/mpt-1b-redpajama-200b'
    # flamingo_checkpoint_path = (
    #     '/data/share/pyz/checkpoint/openflamingo/OpenFlamingo-3B-vitl-mpt1b'
    # )
    # hf_root = 'OpenFlamingo-3B-vitl-mpt1b'
    # cross_attn_every_n_layers = 1

    precision = 'bf16'
    device = 'cuda'
    gen_args = {
        'max_new_tokens': 20,
        'num_beams': 3,
        'length_penalty': 0.0,
        'min_new_tokens': 0,
    }
    other_save_field = ['image_id', 'single_caption', 'captions']

    for data_split in ['train', 'val']:
        image_path = os.path.join(data_root_dir, f'{data_split}2017')
        meta_info_path = os.path.join(image_path, 'metadata.csv')
        if not os.path.exists(meta_info_path) or overwrite:
            ann_path = os.path.join(
                data_root_dir, 'annotations', f'captions_{data_split}2017.json'
            )
            data_dict = load_coco_dict(image_path, ann_path)
            pd.DataFrame(data_dict).to_csv(meta_info_path, index=False)

    ice_prompt = PromptTemplate(
        template='</E><image>Output:<X>',
        ice_token='</E>',
        column_token_map={'single_caption': '<X>'},
    )

    ds = load_dataset("imagefolder", data_dir=data_root_dir)
    ds['validation'] = ds['validation'].select(range(test_data_num))

    dr = DatasetReader(
        ds, input_columns=['single_caption'], output_column='single_caption'
    )

    total_res = {}

    # iclm test
    if test_iclm:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        lm_config = GPT2Config(
            vocab_size=118287,
            n_layer=2,
            n_embd=768,
            n_head=16,
        )
        iclm_model = SenEncodeCaptionICLM(lm_config)
        iclm_model.load_state_dict(
            torch.load(
                'result/model_cpk/sentence_encode_iclm/last-val_loss:11.4405.pth'
            )['model']
        )

        ice_input = None
        ice_idx_seq = []
        with torch.inference_mode():
            iclm_model.eval()
            iclm_model.to(device)
            for d in tqdm(ds['validation']):
                next_ice_idx_seq = []
                next_ice_input = []
                x_input = bert_tokenizer(
                    d['single_caption'], padding=True, return_tensors='pt'
                ).to(device)
                for i in range(4):
                    if ice_input:
                        ice_input = {
                            k: v.unsqueeze(0).to(device) for k, v in ice_input.items()
                        }
                    outputs = iclm_model(x_input, ice_input)
                    next_token_idx = torch.softmax(
                        outputs.logits[:, -1, :], dim=-1
                    ).argmax(dim=-1)
                    next_ice_idx_seq.append(next_token_idx.item())
                    next_ice_input.append(
                        ds['train'][next_token_idx]['single_caption'][0]
                    )
                    ice_input = bert_tokenizer(
                        next_ice_input, padding=True, return_tensors='pt'
                    ).to(device)
                ice_idx_seq.append(next_ice_idx_seq)
        del iclm_model, bert_tokenizer
        gc.collect()
        
        retriever = DirRetriever(
            dr,
            ice_idx_seq,
            ice_separator='<|endofchunk|>',
            ice_eos_token='<|endofchunk|>',
            prompt_eos_token='',
            ice_num=4,
            test_split='validation',
        )
        model, image_processor, tokenizer, autocast_context = init_flamingo(
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            flamingo_checkpoint_path=flamingo_checkpoint_path,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            hf_root=hf_root,
            precision=precision,
            device=device,
        )
        inferencer = FlamingoGenInferencer(
            model,
            tokenizer,
            image_processor,
            other_save_field=other_save_field,
            autocast_context=autocast_context,
            image_field="image",
            batch_size=16,
            generation_kwargs=gen_args,
        )
        total_res[f'{type(retriever).__name__}'] = get_cider(
            inferencer, retriever, ice_prompt, 4
        )
        print(total_res)

    # # zero-shot test
    # if teat_zero_shot:
    #     retriever = ZeroRetriever(
    #         dr,
    #         prompt_eos_token='',
    #         test_split='validation',
    #     )

    #     total_res[f'{type(retriever).__name__}'] = get_cider(
    #         inferencer, retriever, ice_prompt, 0
    #     )
    #     print(total_res)

    # # random sample test
    # if test_random:
    #     single_retriever_res = {}
    #     for shot_num in [4]:
    #         retriever = RandomRetriever(
    #             dr,
    #             ice_separator='<|endofchunk|>',
    #             ice_eos_token='<|endofchunk|>',
    #             prompt_eos_token='',
    #             ice_num=shot_num,
    #             test_split='validation',
    #         )
    #         single_retriever_res[f'{shot_num=}'] = get_cider(
    #             inferencer, retriever, ice_prompt, shot_num
    #         )
    #     total_res[f'{type(retriever).__name__}'] = single_retriever_res
    #     print(total_res)

    # # ICLM sample test
    # if test_iclm:
    #     single_retriever_res = {}
    #     iclm_model = torch.load(
    #         iclm_model_path,
    #         map_location='cpu',
    #     )
    #     test_emb_map = torch.load(test_emb_path)
    #     retriever = ICLMRetriever(
    #         dr,
    #         iclm_model=iclm_model,
    #         test_emb_map=test_emb_map,
    #         ice_separator='<|endofchunk|>',
    #         ice_eos_token='<|endofchunk|>',
    #         prompt_eos_token='',
    #         test_split='validation',
    #     )
    #     for shot_num in [2, 4, 8]:
    #         retriever.ice_num = shot_num
    #         single_retriever_res[f'{shot_num=}'] = get_cider(
    #             inferencer, retriever, ice_prompt, shot_num
    #         )
    #     total_res[f'{type(retriever).__name__}'] = single_retriever_res
    #     print(total_res)

    # if test_topk_caption:
    #     single_retriever_res = {}
    #     retriever = TopkRetriever(
    #         dr,
    #         ice_separator='<|endofchunk|>',
    #         ice_eos_token='<|endofchunk|>',
    #         test_split='validation',
    #     )
    #     for shot_num in [2, 4, 8]:
    #         retriever.ice_num = shot_num
    #         single_retriever_res[f'{shot_num=}'] = get_cider(
    #             inferencer, retriever, ice_prompt, shot_num
    #         )
    #     total_res[f'{type(retriever).__name__}'] = single_retriever_res
    #     print(total_res)

    with open(result_json_path, 'w') as f:
        json.dump(total_res, f, indent=4)
