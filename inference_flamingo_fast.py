import datetime
import json
import logging
import os
import random
import uuid
from typing import Union

import datasets
import hydra
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
from openicl import (
    DatasetReader,
    DirRetriever,
    FlamingoGenInferencerFast,
    MMTopkRetriever,
    PromptTemplate,
    RandomRetriever,
    ZeroRetriever,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from src.load_ds_utils import load_coco_ds, load_vqav2_ds
from src.metrics.cider_calculator import compute_cider
from src.metrics.vqa_metrics import compute_vqa_accuracy, postprocess_vqa_generation
from src.models import GPT2ICLM, LSTMICLM
from src.utils import init_flamingo

logger = logging.getLogger(__name__)


def record(result_json_path: str, new_data: dict):
    recorded_data = {}
    if os.path.exists(result_json_path):
        with open(result_json_path, 'r') as f:
            recorded_data = json.load(f)

    with open(result_json_path, 'w') as f:
        recorded_data.update(new_data)
        json.dump(recorded_data, f, indent=4)


def evaluate_retriever(
    retriever_name,
    inferencer,
    retriever,
    ice_prompt,
    base_info,
    shot_num_list,
    result_json_path,
    cfg,
):
    retriever_res = {}
    info = base_info + retriever_name
    for shot_num in shot_num_list:
        logger.info(
            f'Now begin test {cfg.task.task_name}: {retriever_name} with {shot_num=}'
        )
        output_files = info + f'-bs:{cfg.inference_bs}-{shot_num=}'
        retriever.ice_num = shot_num
        if cfg.task.task_name == 'caption':
            metric = inference_caption(
                inferencer,
                retriever,
                ice_prompt,
                cfg.dataset.val_coco_annotation_file,
                output_files,
            )
        elif cfg.task.task_name == 'vqa':
            metric = inference_vqa(
                inferencer=inferencer,
                retriever=retriever,
                ice_prompt=ice_prompt,
                val_ques_path=cfg.dataset.val_ques_path,
                val_ann_path=cfg.dataset.val_ann_path,
                output_json_filename=output_files,
            )
        retriever_res[f'{shot_num=}'] = metric
        logger.info(f'{output_files}: {metric=}')
        record(result_json_path, {info: retriever_res})


def init_retriever(retriever_name, dr, cfg):
    if retriever_name == 'ZeroShot':
        return ZeroRetriever(dr, prompt_eos_token='', test_split='validation')
    elif retriever_name == 'RandomSample':
        return RandomRetriever(
            dr,
            ice_separator='<|endofchunk|>',
            ice_eos_token='<|endofchunk|>',
            test_split='validation',
            seed=cfg.seed,
        )
    elif retriever_name.startswith('MMTopKRetriever'):
        mode = retriever_name.split('-')[-1]
        index_field = (
            cfg.task.ice_text_feature_field
            if mode.endswith('t')
            else cfg.task.image_field
        )
        test_field = (
            cfg.task.image_field
            if mode.startswith('i')
            else cfg.task.ice_text_feature_field
        )

        cache_file = os.path.join(
            cfg.result_dir,
            'cache',
            f'{cfg.task.task_name}-{cfg.dataset.name}-{cfg.mmtopk_clip_name.split("/")[-1]}-{mode}-'
            f'index_field:{index_field}-test_data_num:{cfg.test_data_num}-'
            f'test_field:{test_field}-emb_cache.pth',
        )
        return MMTopkRetriever(
            dr,
            ice_separator='<|endofchunk|>',
            ice_eos_token='<|endofchunk|>',
            test_split='validation',
            batch_size=32,
            num_workers=8,
            mode=mode,
            index_field=index_field,
            test_field=test_field,
            clip_model_name=cfg.mmtopk_clip_name,
            cache_file=cache_file,
        )
    # Add other retrievers if needed
    return None


def inference_caption(
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
    cider_score = compute_cider(pred_coco, val_ann_path)
    return cider_score * 100


def inference_vqa(
    inferencer, retriever, ice_prompt, val_ques_path, val_ann_path, output_json_filename
):
    output_dict = inferencer.inference(
        retriever,
        ice_prompt,
        output_json_filename=output_json_filename,
        return_dict=True,
    )
    preds = []
    for idx in output_dict:
        preds.append(
            {
                'answer': postprocess_vqa_generation(output_dict[idx]['prediction']),
                'question_id': output_dict[idx]['question_id'],
            }
        )
    random_uuid = str(uuid.uuid4())

    with open(f'{random_uuid}.json', 'w') as f:
        f.write(json.dumps(preds, indent=4))
    acc = compute_vqa_accuracy(f"{random_uuid}.json", val_ques_path, val_ann_path)
    # delete the temporary file
    os.remove(f"{random_uuid}.json")
    return acc


@hydra.main(
    version_base=None, config_path="./configs", config_name="inference_fast.yaml"
)
def main(cfg: DictConfig):
    logger.info(f'{cfg=}')
    result_dir = os.path.join(
        cfg.result_dir,
        'flamingo_inference',
        cfg.task.task_name,
        cfg.ex_name,
    )
    result_json_path = os.path.join(result_dir, 'metrics.json')

    ice_prompt = PromptTemplate(
        template=cfg.task.template,
        ice_token=cfg.task.ice_token,
        column_token_map=dict(cfg.task.column_token_map),
    )
    test_data_num = cfg.test_data_num
    index_data_num = cfg.index_data_num
    if cfg.task.task_name == 'caption':
        ds = load_coco_ds(cfg)
    elif cfg.task.task_name == 'vqa':
        ds = load_vqav2_ds(cfg)
    else:
        raise ValueError(f'{cfg.task.task_name=} error, should in ["caption", "vqa"]')

    test_split = 'validation'
    if index_data_num != -1:
        ds['train'] = ds['train'].select(
            random.sample(range(len(ds['train'])), index_data_num)
        )
    if test_data_num != -1:
        ds[test_split] = ds[test_split].select(range(test_data_num))

    dr = DatasetReader(
        ds,
        input_columns=list(cfg.task.input_columns),
        output_column=cfg.task.output_column,
    )

    model, image_processor, tokenizer, autocast_context = init_flamingo(
        lang_encoder_path=cfg.flamingo.lang_encoder_path,
        tokenizer_path=cfg.flamingo.tokenizer_path,
        flamingo_checkpoint_dir=cfg.flamingo.flamingo_checkpoint_dir,
        cross_attn_every_n_layers=cfg.flamingo.cross_attn_every_n_layers,
        hf_root=cfg.flamingo.hf_root,
        precision=cfg.precision,
        device=cfg.device,
        from_local=cfg.flamingo.load_from_local,
    )
    inferencer = FlamingoGenInferencerFast(
        model,
        tokenizer,
        image_processor,
        other_save_field=cfg.task.other_save_field,
        autocast_context=autocast_context,
        image_field=cfg.task.image_field,
        batch_size=cfg.inference_bs,
        num_workers=cfg.num_workers,
        num_proc=cfg.num_proc,
        preprocessor_bs=cfg.preprocessor_bs,
        generation_kwargs=cfg.task.gen_args,
        output_json_filepath=os.path.join(result_dir, 'generation_metainfo'),
    )

    base_info = f'{str(datetime.datetime.now())}-{test_data_num=}-'

    retriever_list = [
        ('ZeroShot', [0] if cfg.test_zero_shot else []),
        ('RandomSample', cfg.shot_num_list if cfg.test_random else []),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-i2t',
            cfg.shot_num_list if cfg.test_i2t else [],
        ),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-i2i',
            cfg.shot_num_list if cfg.test_i2i else [],
        ),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-t2t',
            cfg.shot_num_list if cfg.test_t2t else [],
        ),
    ]

    # Test for other
    for retriever_name, shot_nums in retriever_list:
        if shot_nums:  # Only initialize and evaluate if shot_nums is not empty
            retriever_instance = init_retriever(retriever_name, dr, cfg)
            evaluate_retriever(
                retriever_name,
                inferencer,
                retriever_instance,
                ice_prompt,
                base_info,
                shot_nums,
                result_json_path,
                cfg,
            )
    # ICLM sample test
    if cfg.test_iclm:
        retriever_res = {}
        iclm_model = hydra.utils.instantiate(cfg.train.iclm_model)
        if cfg.iclm_path is None:
            logger.info(
                f'detect iclm_path is None, now try to find in model_cpk/{cfg.ex_name}'
            )
            cpk_dir = os.path.join(cfg.result_dir, 'model_cpk', cfg.ex_name)
            cpk_list = []
            for f in os.listdir(cpk_dir):
                cpk_list.append(os.path.join(cpk_dir, f))
            cpk_list = list(filter(lambda x: 'last' in x, cpk_list))
            if cpk_list:
                logger.info(f'Detect {cpk_list[0]}, now begin to load cpk...')
                iclm_path = cpk_list[0]
            else:
                raise ValueError(
                    f'The iclm_path is None and detect no checkpoint can use in {cpk_dir}'
                )
        else:
            iclm_path = cfg.iclm_path
        iclm_model.load_state_dict(torch.load(iclm_path)['model'])

        processor = AutoProcessor.from_pretrained(cfg.train.iclm_model.clip_name)
        retriever_info = 'ICLM-' + os.path.splitext(os.path.basename(iclm_path))[0]

        info = (
            base_info
            + retriever_info
            + f'-{iclm_model.query_encoding_flag=}-{iclm_model.ice_encoding_flag=}-bs:{cfg.inference_bs}'
        )

        ice_idx_list = iclm_generation(
            iclm_model=iclm_model,
            val_ds=ds[test_split],
            train_ds=ds['train'],
            processor=processor,
            shot_num=max(cfg.shot_num_list),
            cfg=cfg,
        )

        for shot_num in cfg.shot_num_list:
            logger.info(f'Now begin test: {retriever_info} with {shot_num=}')
            output_files = info + f'-bs:{cfg.inference_bs}-{shot_num=}'
            need_ice_idx_list = [ice_idx[:shot_num] for ice_idx in ice_idx_list]
            if cfg.random_order_iclm_ice:
                need_ice_idx_list = shuffle_2d_list(need_ice_idx_list)
            retriever = DirRetriever(
                dr,
                need_ice_idx_list,
                ice_separator='<|endofchunk|>',
                ice_eos_token='<|endofchunk|>',
                prompt_eos_token='',
                test_split=test_split,
            )
            retriever_info = 'ICLM'
            retriever.ice_num = shot_num
            if cfg.task.task_name == 'caption':
                metric = inference_caption(
                    inferencer,
                    retriever,
                    ice_prompt,
                    cfg.dataset.val_coco_annotation_file,
                    output_files,
                )
            elif cfg.task.task_name == 'vqa':
                metric = inference_vqa(
                    inferencer=inferencer,
                    retriever=retriever,
                    ice_prompt=ice_prompt,
                    val_ques_path=cfg.dataset.val_ques_path,
                    val_ann_path=cfg.dataset.val_ann_path,
                    output_json_filename=output_files,
                )
            retriever_res[f'{shot_num=}'] = metric
            logger.info(f'{output_files}: {metric=}')
            record(result_json_path, {info: retriever_res})


def shuffle_2d_list(matrix):
    new_matrix = [row.copy() for row in matrix]
    if len(new_matrix[0]) == 1:
        return new_matrix
    for i, row in enumerate(tqdm(new_matrix)):

        while row == matrix[i]:
            random.shuffle(row)
    return new_matrix


@torch.inference_mode()
def idx_iclm_generation(iclm_model, ds, img_processor, shot_num, device, eos_token_id):
    iclm_model = iclm_model.to(device)
    iclm_model.eval()
    ice_idx_list = []
    bos_token_id = eos_token_id + 1
    query_token_id = eos_token_id + 2
    init_ice_idx = torch.tensor([[bos_token_id, query_token_id]]).to(device)

    for data in tqdm(ds, ncols=100):
        img = data['image']
        img = img_processor(images=img, return_tensors='pt').to(device)['pixel_values']

        num_beams = 10
        if shot_num == 1:
            num_beams = 1

        res = iclm_model.generation(
            img,
            init_ice_idx=init_ice_idx,
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
                init_ice_idx=init_ice_idx,
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


@torch.inference_mode()
def iclm_generation(
    iclm_model: Union[GPT2ICLM, LSTMICLM],
    val_ds: datasets.Dataset,
    train_ds: datasets.Dataset,
    processor,
    shot_num,
    cfg,
):
    iclm_model = iclm_model.to(cfg.device)
    iclm_model.eval()
    ice_idx_list = []
    bos_token_id = len(train_ds) + 1
    query_token_id = len(train_ds) + 2

    query_image_field = cfg.train.iclm_ds.query_image_field
    query_text_field = cfg.train.iclm_ds.query_text_field

    val_ds_ = val_ds.map()

    def prepare(examples):
        images = texts = None
        if query_image_field:
            images = [i for i in examples[query_image_field]]
        if query_text_field:
            texts = [i for i in examples[query_text_field]]

        data_dict = processor(
            images=images,
            text=texts,
            padding=True,
            return_tensors="pt",
        )
        return data_dict

    val_ds_.set_transform(prepare)
    dataloader = DataLoader(
        val_ds_,
        batch_size=cfg.iclm_bs,
        shuffle=False,
        num_workers=cfg.iclm_num_workers,
    )

    for query_input in tqdm(dataloader, ncols=100):
        query_input = {k: v.to(cfg.device) for k, v in query_input.items()}
        bs = len(query_input[list(query_input.keys())[0]])
        init_ice_idx = torch.tensor(
            [[bos_token_id, query_token_id] for _ in range(bs)]
        ).to(cfg.device)
        res = iclm_model.generation(
            query_input=query_input,
            init_ice_idx=init_ice_idx,
            shot_num=shot_num,
            index_ds=train_ds,
            processor=processor,
            ice_image_field=cfg.train.iclm_ds.ice_image_field,
            ice_text_field=cfg.train.iclm_ds.ice_text_field,
            device=cfg.device,
        )
        res = [r[2 : 2 + shot_num] for r in res]
        ice_idx_list.extend(res)

    return ice_idx_list


if __name__ == '__main__':
    load_dotenv()
    main()
