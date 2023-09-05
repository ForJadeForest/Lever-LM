import datetime
import json
import logging
import os

import hydra
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from openicl import (
    DatasetReader,
    DirRetriever,
    FlamingoGenInferencer,
    MMTopkRetriever,
    PromptTemplate,
    RandomRetriever,
    TopkRetriever,
    ZeroRetriever,
)
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from datasets import load_dataset
from src.datasets import CocoDataset
from src.metrics.cider_utils import compute_cider
from src.utils import init_flamingo, load_karpathy_split
from src.models import SenImgEncodeCaptionICLM, IdxBaseCaptionICLM

logger = logging.getLogger(__name__)


def construct_coco_dict(coco_root, version=2014, overwrite=False):
    version_coco_root = os.path.join(coco_root, f'mscoco{version}')
    for data_split in ['train', 'val']:
        split_image_path = os.path.join(version_coco_root, f'{data_split}{version}')
        meta_info_path = os.path.join(split_image_path, 'metadata.csv')
        if not os.path.exists(meta_info_path) or overwrite:
            ann_path = os.path.join(
                version_coco_root, 'annotations', f'captions_{data_split}{version}.json'
            )
            dataset = CocoDataset(split_image_path, ann_path)

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
            pd.DataFrame(data_dict).to_csv(meta_info_path, index=False)


def inference_cider(
    inferencer,
    retriever,
    ice_prompt,
    val_ann_path,
    output_json_filename,
):
    output_dict = inferencer.inference(
        retriever,
        ice_prompt,
        output_json_filename=output_json_filename,
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
    cider_score = compute_cider(pred_coco, val_ann_path)['CIDEr']
    return cider_score


@hydra.main(version_base=None, config_path="./configs", config_name="inference.yaml")
def main(cfg: DictConfig):
    construct_coco_dict(
        cfg.dataset.coco_root, cfg.dataset.version, cfg.overwrite_metainfo
    )

    result_json_path = os.path.join(
        cfg.result_dir, 'flamingo_inference', cfg.ex_name, 'total_cider_res.json'
    )
    ice_prompt = PromptTemplate(
        template='</E><image>Output:<X>',
        ice_token='</E>',
        column_token_map={'single_caption': '<X>'},
    )

    test_data_num = cfg.test_data_num
    if cfg.use_karpathy_split:
        ds = load_karpathy_split(cfg)
        if test_data_num != -1:
            ds['test'] = ds['test'].select(range(test_data_num))
        test_split = 'test'
    else:
        ds = load_dataset("imagefolder", data_dir=cfg.dataset.coco_version_root)
        if test_data_num != -1:
            ds['validation'] = ds['validation'].select(range(test_data_num))
        test_split = 'validation'

    dr = DatasetReader(
        ds, input_columns=['single_caption'], output_column='single_caption'
    )

    total_cider_res = {}
    model, image_processor, tokenizer, autocast_context = init_flamingo(
        lang_encoder_path=cfg.flamingo.lang_encoder_path,
        tokenizer_path=cfg.flamingo.tokenizer_path,
        flamingo_checkpoint_path=cfg.flamingo.flamingo_checkpoint_path,
        cross_attn_every_n_layers=cfg.flamingo.cross_attn_every_n_layers,
        hf_root=cfg.flamingo.hf_root,
        precision=cfg.precision,
        device=cfg.device,
    )
    inferencer = FlamingoGenInferencer(
        model,
        tokenizer,
        image_processor,
        other_save_field=cfg.other_save_field,
        autocast_context=autocast_context,
        image_field="image",
        batch_size=cfg.inference_bs,
        generation_kwargs=cfg.gen_args,
        output_json_filepath=os.path.join(
            cfg.result_dir, 'flamingo_inference', cfg.ex_name, 'generation_metainfo'
        ),
    )
    # zero-shot test
    if cfg.test_zero_shot:
        retriever = ZeroRetriever(
            dr,
            prompt_eos_token='',
            test_split=test_split,
        )
        shot_num = 0
        output_files = (
            f'{str(datetime.datetime.now())}-{type(inferencer).__name__}-'
            f'{type(retriever).__name__}-{shot_num=}-{test_data_num=}'
        )

        cider_score = inference_cider(
            inferencer,
            retriever,
            ice_prompt,
            cfg.dataset.val_coco_annotation_file,
            output_files,
        )
        total_cider_res[f'{type(retriever).__name__}'] = cider_score

        logger.info(
            f'{type(retriever).__name__}-{shot_num=}-{test_data_num=}-{cider_score=}'
        )
        logger.debug(total_cider_res)

        with open(result_json_path, 'w') as f:
            json.dump(total_cider_res, f, indent=4)

    if cfg.test_random:
        single_retriever_res = {}
        retriever = RandomRetriever(
            dr,
            ice_separator='<|endofchunk|>',
            ice_eos_token='<|endofchunk|>',
            test_split=test_split,
        )
        for shot_num in cfg.shot_num_list:
            output_files = (
                f'{str(datetime.datetime.now())}-{type(inferencer).__name__}-'
                f'{type(retriever).__name__}-{shot_num=}-{test_data_num=}'
            )
            retriever.ice_num = shot_num
            cider_score = inference_cider(
                inferencer,
                retriever,
                ice_prompt,
                cfg.dataset.val_coco_annotation_file,
                output_files,
            )
            single_retriever_res[f'{shot_num=}'] = cider_score
            logger.info(f'{type(retriever).__name__}-{shot_num=}-{cider_score=}')
        total_cider_res[f'{type(retriever).__name__}'] = single_retriever_res
        logger.debug(total_cider_res)

        with open(result_json_path, 'w') as f:
            json.dump(total_cider_res, f, indent=4)

    if cfg.test_i2t:
        single_retriever_res = {}
        retriever = MMTopkRetriever(
            dr,
            ice_separator='<|endofchunk|>',
            ice_eos_token='<|endofchunk|>',
            test_split=test_split,
            batch_size=32,
            mode='i2t',
            index_field='single_caption',
            test_field='image',
            clip_model_name=cfg.mmtopk_clip_name,
        )

        for shot_num in cfg.shot_num_list:
            output_files = (
                f'{str(datetime.datetime.now())}-{type(inferencer).__name__}-'
                f'{type(retriever).__name__}-{shot_num=}-{test_data_num=}-'
                f'{cfg.mmtopk_clip_name.replace("/", "-")}-i2t'
            )

            retriever.ice_num = shot_num
            cider_score = inference_cider(
                inferencer,
                retriever,
                ice_prompt,
                cfg.dataset.val_coco_annotation_file,
                output_files,
            )
            single_retriever_res[f'{shot_num=}'] = cider_score
            logger.info(f'{type(retriever).__name__}-{shot_num=}-{cider_score=}')

        total_cider_res[
            f'{type(retriever).__name__}-i2t-{cfg.mmtopk_clip_name}'
        ] = single_retriever_res

        logger.debug(total_cider_res)
        with open(result_json_path, 'w') as f:
            json.dump(total_cider_res, f, indent=4)

    if cfg.test_i2i:
        single_retriever_res = {}
        retriever = MMTopkRetriever(
            dr,
            ice_separator='<|endofchunk|>',
            ice_eos_token='<|endofchunk|>',
            test_split=test_split,
            batch_size=32,
            mode='i2i',
            index_field='image',
            test_field='image',
            clip_model_name=cfg.mmtopk_clip_name,
        )

        for shot_num in cfg.shot_num_list:
            output_files = (
                f'{str(datetime.datetime.now())}-{type(inferencer).__name__}-'
                f'{type(retriever).__name__}-{shot_num=}-{test_data_num=}-'
                f'{cfg.mmtopk_clip_name.replace("/", "-")}-i2i'
            )

            retriever.ice_num = shot_num
            cider_score = inference_cider(
                inferencer,
                retriever,
                ice_prompt,
                cfg.dataset.val_coco_annotation_file,
                output_files,
            )
            single_retriever_res[f'{shot_num=}'] = cider_score
            logger.info(f'{type(retriever).__name__}-{shot_num=}-{cider_score=}')

        total_cider_res[
            f'{type(retriever).__name__}-i2i-{cfg.mmtopk_clip_name}'
        ] = single_retriever_res

        logger.debug(total_cider_res)
        with open(result_json_path, 'w') as f:
            json.dump(total_cider_res, f, indent=4)

    if cfg.test_topk_caption:
        single_retriever_res = {}
        retriever = TopkRetriever(
            dr,
            ice_separator='<|endofchunk|>',
            ice_eos_token='<|endofchunk|>',
            test_split=test_split,
            batch_size=512,
        )
        for shot_num in cfg.shot_num_list:
            output_files = f'{str(datetime.datetime.now())}-{type(inferencer).__name__}-{type(retriever).__name__}-{shot_num=}-{test_data_num=}'
            retriever.ice_num = shot_num
            cider_score = inference_cider(
                inferencer,
                retriever,
                ice_prompt,
                cfg.dataset.val_coco_annotation_file,
                output_files,
            )
            single_retriever_res[f'{shot_num=}'] = cider_score
            logger.info(f'{type(retriever).__name__}-{shot_num=}-{cider_score=}')
        total_cider_res[f'{type(retriever).__name__}'] = single_retriever_res
        logger.debug(total_cider_res)
        with open(result_json_path, 'w') as f:
            json.dump(total_cider_res, f, indent=4)

    # ICLM sample test
    if cfg.test_iclm:
        single_retriever_res = {}
        iclm_model = hydra.utils.instantiate(cfg.train.iclm_model)
        iclm_model.load_state_dict(torch.load(cfg.iclm_path)['model'])

        image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        for shot_num in cfg.shot_num_list:
            if isinstance(iclm_model, SenImgEncodeCaptionICLM):
                ice_idx_list = clip_base_iclm_gene_ice(
                    iclm_model,
                    ds[test_split],
                    ds['train'],
                    image_processor,
                    tokenizer,
                    shot_num,
                    cfg.device,
                )
            elif isinstance(iclm_model, IdxBaseCaptionICLM):
                ice_idx_list = idx_base_iclm_gene_ice(
                    iclm_model,
                    ds[test_split],
                    image_processor,
                    shot_num,
                    cfg.device,
                    cfg.eos_token_id,
                )
            retriever = DirRetriever(
                dr,
                ice_idx_list,
                ice_separator='<|endofchunk|>',
                ice_eos_token='<|endofchunk|>',
                prompt_eos_token='',
                test_split=test_split,
            )
            output_files = f'{str(datetime.datetime.now())}-{type(inferencer).__name__}-ICLMRetriver-{shot_num=}-{test_data_num=}'
            retriever.ice_num = shot_num
            cider_score = inference_cider(
                inferencer,
                retriever,
                ice_prompt,
                cfg.dataset.val_coco_annotation_file,
                output_files,
            )
            single_retriever_res[f'{shot_num=}'] = cider_score
            logger.info(f'{type(retriever).__name__}-{shot_num=}-{cider_score=}')
        total_cider_res['ICLMRetriver'] = single_retriever_res
        logger.debug(total_cider_res)

        with open(result_json_path, 'w') as f:
            json.dump(total_cider_res, f, indent=4)


@torch.inference_mode()
def idx_base_iclm_gene_ice(
    iclm_model, ds, img_processor, shot_num, device, eos_token_id
):
    iclm_model = iclm_model.to(device)
    iclm_model.eval()
    ice_idx_list = []

    for data in tqdm(ds):
        img = data['image']
        img = img_processor(images=img, return_tensors='pt').to(device)['pixel_values']
        ice_input = torch.tensor([[118288, 118289]]).to(device)

        num_beams = 10
        if shot_num == 1:
            num_beams = 1

        res = iclm_model.generation(
            img,
            ice_input=ice_input,
            repetition_penalty=2.0,
            max_new_tokens=shot_num,
            num_beams=num_beams,
            min_length=shot_num,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
        )[0]
        res = res[2 : 2 + shot_num]
        if eos_token_id in res:
            res = iclm_model.generation(
                img,
                ice_input=ice_input,
                repetition_penalty=2.0,
                max_new_tokens=shot_num,
                num_beams=num_beams,
                bad_words_ids=[[eos_token_id]],
                min_length=shot_num,
            )[0]
            res = res[2 : 2 + shot_num]

        assert len(res) == shot_num, f'{len(res)=}'
        assert eos_token_id not in res, f'{res=}'
        ice_idx_list.append(res)
    return ice_idx_list


def clip_base_iclm_gene_ice(
    iclm_model, val_ds, train_ds, img_processor, tokenizer, shot_num, device
):
    iclm_model = iclm_model.to(device)
    iclm_model.eval()
    ice_idx_list = []

    for data in tqdm(val_ds):
        img = data['image']
        img = img_processor(images=img, return_tensors='pt').to(device)['pixel_values']
        res = iclm_model.generation(
            img, shot_num, train_ds, tokenizer, repetition_penalty=2.0
        )[0]
        res = res[2 : 2 + shot_num]
        assert len(res) == shot_num, f'{len(res)=}'
        assert 118287 not in res, f'{res=}'
        ice_idx_list.append(res)
    return ice_idx_list


if __name__ == '__main__':
    load_dotenv()
    main()
