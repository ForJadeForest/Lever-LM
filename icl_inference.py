import datetime
import json
import os
import random
import uuid

import hydra
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoProcessor

from src.metrics.cider_calculator import compute_cider
from src.metrics.vqa_metrics import compute_vqa_accuracy, postprocess_vqa_generation
from src.retriever import *
from src.utils import init_lvlm
from src.vl_icl_inferencer import VLICLInferecer
from utils import caption_postprocess, load_ds, vqa_postprocess


def get_icd_lm_path(cfg):
    if cfg.icd_lm_path is None:
        logger.info(
            f'detect icd_lm_path is None, now try to find in {cfg.result_dir}/model_cpk/{cfg.ex_name}'
        )
        cpk_dir = os.path.join(
            cfg.result_dir, 'model_cpk', cfg.task.task_name, cfg.ex_name
        )
        cpk_list = []
        for f in os.listdir(cpk_dir):
            cpk_list.append(os.path.join(cpk_dir, f))
        cpk_list = list(filter(lambda x: cfg.default_cpk_key in x, cpk_list))
        if cpk_list:
            logger.info(f'Detect {cpk_list[0]}, now begin to load cpk...')
            icd_lm_path = cpk_list[0]
        else:
            raise ValueError(
                f'The icd_lm_path is None and detect no checkpoint can use in {cpk_dir}'
            )
    else:
        icd_lm_path = cfg.icd_lm_path
    return icd_lm_path


def init_icd_lm(cfg, icd_lm_path):
    icd_lm = hydra.utils.instantiate(cfg.train.icd_lm)
    state_dict = torch.load(icd_lm_path)['state_dict']
    state_dict = {k.replace('icd_lm.', ''): v for k, v in state_dict.items()}
    icd_lm.load_state_dict(state_dict)
    processor = AutoProcessor.from_pretrained(cfg.train.icd_lm.clip_name)
    return icd_lm, processor


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
    ds,
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
        icd_idx_list = retriever.retrieve(shot_num)

        if cfg.task.task_name == 'caption':
            metric = inference_caption(
                inferencer=inferencer,
                ds=ds,
                icd_idx_list=icd_idx_list,
                val_ann_path=cfg.dataset.val_coco_annotation_file,
                output_json_filename=output_files,
                model_name=cfg.lvlm.name,
            )
        elif cfg.task.task_name == 'vqa':
            metric = inference_vqa(
                inferencer=inferencer,
                ds=ds,
                icd_idx_list=icd_idx_list,
                val_ques_path=cfg.dataset.val_ques_path,
                val_ann_path=cfg.dataset.val_ann_path,
                output_json_filename=output_files,
                model_name=cfg.lvlm.name,
            )
        retriever_res[f'{shot_num=}'] = metric
        logger.info(f'{output_files}: {metric=}')
        record(result_json_path, {info: retriever_res})


def init_retriever(retriever_name, ds, cfg):
    if retriever_name == 'ZeroShot':
        return ZeroRetriever(ds['train'], ds['validation'])
    elif retriever_name == 'RandomRetriever':
        return RandRetriever(
            ds['train'],
            ds['validation'],
            seed=cfg.seed,
            fixed=cfg.random_retrieval_fixed,
        )
    elif retriever_name.startswith('MMTopKRetriever'):
        mode = retriever_name.split('-')[-1]
        index_field = (
            cfg.task.icd_text_feature_field
            if mode.endswith('t')
            else cfg.task.image_field
        )
        test_field = (
            cfg.task.image_field
            if mode.startswith('i')
            else cfg.task.icd_text_feature_field
        )

        cache_file = os.path.join(
            cfg.result_dir,
            'cache',
            f'{cfg.task.task_name}-{cfg.dataset.name}-{cfg.mmtopk_clip_name.split("/")[-1]}-{mode}-'
            f'index_field:{index_field}-test_data_num:{cfg.test_data_num}-'
            f'test_field:{test_field}-emb_cache.pth',
        )
        return MMTopkRetriever(
            ds['train'],
            ds['validation'],
            mode=mode,
            index_field=index_field,
            test_field=test_field,
            clip_model_name=cfg.mmtopk_clip_name,
            cache_file=cache_file,
            reversed_order=cfg.mmtopk_reversed_order,
            batch_size=32,
            num_workers=8,
        )
    elif retriever_name == 'ICDLMRetriever':
        icd_lm_path = get_icd_lm_path(cfg)
        icd_lm, processor = init_icd_lm(cfg, icd_lm_path=icd_lm_path)
        return ICDLMRetriever(
            ds['train'],
            ds['validation'],
            icd_lm=icd_lm,
            processor=processor,
            query_image_field=cfg.train.icd_lm_ds.query_image_field,
            query_text_field=cfg.train.icd_lm_ds.query_text_field,
            icd_image_field=cfg.train.icd_lm_ds.icd_image_field,
            icd_text_field=cfg.train.icd_lm_ds.icd_text_field,
            device=cfg.device,
            infer_batch_size=cfg.icd_lm_bs,
            infer_num_workers=cfg.icd_lm_num_workers,
            reverse_seq=cfg.reverse_seq,
        )

    return None


def inference_caption(
    inferencer,
    ds,
    icd_idx_list,
    val_ann_path,
    output_json_filename,
    model_name,
):
    output_dict = inferencer.inference(
        train_ds=ds['train'],
        test_ds=ds['validation'],
        ice_idx_list=icd_idx_list,
        output_json_filename=output_json_filename,
    )
    pred_coco = []
    for idx in output_dict:
        pred_coco.append(
            {
                'image_id': output_dict[idx]['image_id'],
                'caption': caption_postprocess(
                    output_dict[idx]['prediction'], model_name
                ),
            }
        )
    cider_score = compute_cider(pred_coco, val_ann_path)
    return cider_score * 100


def inference_vqa(
    inferencer,
    ds,
    icd_idx_list,
    val_ques_path,
    val_ann_path,
    output_json_filename,
    model_name,
):
    output_dict = inferencer.inference(
        train_ds=ds['train'],
        test_ds=ds['validation'],
        ice_idx_list=icd_idx_list,
        output_json_filename=output_json_filename,
    )
    preds = []
    for idx in output_dict:
        preds.append(
            {
                'answer': vqa_postprocess(
                    output_dict[idx]['prediction'], model_name=model_name
                ),
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


@hydra.main(version_base=None, config_path="./configs", config_name="inference.yaml")
def main(cfg: DictConfig):
    logger.info(f'{cfg=}')
    result_dir = os.path.join(
        cfg.result_dir,
        'icl_inference',
        cfg.lvlm.name,
        cfg.task.task_name,
        cfg.ex_name,
    )
    result_json_path = os.path.join(result_dir, 'metrics.json')

    test_data_num = cfg.test_data_num
    index_data_num = cfg.index_data_num

    ds = load_ds(cfg)

    if index_data_num != -1:
        ds['train'] = ds['train'].select(
            random.sample(range(len(ds['train'])), index_data_num)
        )
    if test_data_num != -1:
        ds['validation'] = ds['validation'].select(range(test_data_num))

    interface = init_lvlm(cfg, device=cfg.device)

    inferencer = VLICLInferecer(
        interface=interface,
        train_ds=ds['train'],
        test_ds=ds['validation'],
        generation_kwargs=cfg.task.gen_args,
        other_save_field=cfg.task.other_save_field,
        num_workers=cfg.num_workers,
        num_proc=cfg.num_proc,
        batch_size=cfg.inference_bs,
        output_json_filepath=os.path.join(result_dir, 'generation_metainfo'),
    )

    base_info = f'{str(datetime.datetime.now())}-{test_data_num=}-'

    retriever_list = [
        ('ZeroShot', [0] if cfg.test_zero_shot else []),
        ('RandomRetriever', cfg.shot_num_list if cfg.test_random else []),
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
        (
            'ICDLMRetriever',
            cfg.shot_num_list if cfg.test_icd_lm else [],
        ),
    ]

    # Test for other
    for retriever_name, shot_nums in retriever_list:
        if shot_nums:  # Only initialize and evaluate if shot_nums is not empty
            retriever_instance = init_retriever(retriever_name, ds, cfg)
            evaluate_retriever(
                retriever_name,
                inferencer,
                retriever_instance,
                ds,
                base_info,
                shot_nums,
                result_json_path,
                cfg,
            )


def shuffle_2d_list(matrix):
    new_matrix = [row.copy() for row in matrix]
    if len(new_matrix[0]) == 1:
        return new_matrix
    for i, row in enumerate(tqdm(new_matrix)):
        while row == matrix[i]:
            random.shuffle(row)
    return new_matrix


if __name__ == '__main__':
    load_dotenv()
    main()
