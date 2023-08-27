import argparse
import json
import os
import random
from time import sleep
from typing import Dict, List

import more_itertools
import torch
from PIL import Image
from torch.multiprocessing import spawn
from tqdm import tqdm

from src.datasets import CocoDataset
from src.info_score import get_info_score
from src.utils import encode_image, encode_text, init_flamingo, recall_sim_feature


def get_caption_prompt(caption=None) -> str:
    return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name. Currently only `OpenFlamingo` is supported.",
        default="open_flamingo",
    )
    # coco dataset args
    parser.add_argument(
        '--train_coco_dataset_root', type=str, help='The train2017 coco dataset root'
    )
    parser.add_argument(
        '--train_coco_annotation_file',
        type=str,
        help='The train2017 coco dataset annotation file',
    )
    # open_flamingo args
    parser.add_argument("--lang_encoder_path", type=str, help="The lang_encoder_path")
    parser.add_argument('--tokenizer_path', type=str, help='the tokenizer_path ')
    parser.add_argument(
        '--flamingo_checkpoint_path', type=str, help='The checkpoint of open_flamingo'
    )
    parser.add_argument('--cross_attn_every_n_layers', type=int, help='the ')
    parser.add_argument('--hf_root', type=str, help='the flamingo version')

    # get query_set args
    parser.add_argument(
        '--sim_method',
        type=str,
        default='caption',
        help="the method of getting the small dataset",
    )
    parser.add_argument(
        '--sim_model_type',
        type=str,
        default='openai/clip-vit-large-patch14',
        help="the model type of encoding the dataset",
    )
    parser.add_argument(
        '--query_set_batch_size',
        type=int,
        default=64,
        help="the batch_size of encoding the dataset",
    )
    parser.add_argument(
        '--query_top_k',
        type=int,
        default=200,
        help='the topk nearest as the candidates',
    )

    # generation args
    parser.add_argument(
        '--beam_size',
        type=int,
        default=5,
        help='the beam size generate the new dataset',
    )
    parser.add_argument(
        '--few_shot_num',
        type=int,
        default=5,
        help='the few-shot num of generated dataset ',
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help='the batch size in inference'
    )
    parser.add_argument("--device", type=str, default='cuda', help="model device")
    parser.add_argument("--precision", type=str, default='bf16', help="model device")
    parser.add_argument(
        '--sample_num',
        type=int,
        default=5000,
        help="the random sample num from train dataset",
    )
    # Others
    parser.add_argument(
        '--result_dir', type=str, default=None, help="JSON file to save results"
    )
    parser.add_argument(
        '--gpu_ids', nargs='+', help='The GPU ids you want to use', required=True
    )

    return parser.parse_args()


def load_feature_cache(args, cache_path, encoding_method, coco_dataset, data_key):
    if os.path.exists(cache_path):
        features = torch.load(cache_path)
    else:
        data_list = [d[data_key] for d in coco_dataset]
        features = encoding_method(
            data_list, args.device, args.sim_model_type, args.query_set_batch_size
        )
        torch.save(features, cache_path)
    return features


def beam_filter(score_list, data_id_list, beam_size):
    score_list = torch.tensor(score_list)
    score_value, indices = torch.topk(score_list, beam_size)
    return score_value.tolist(), [data_id_list[idx] for idx in indices]


@torch.inference_mode()
def generate_single_sample_ice(
    model,
    tokenizer,
    image_processor,
    test_data: Dict,
    candidate_set: List[Dict],
    autocast_context,
    device,
    few_shot_num=4,
    batch_size=32,
    beam_size=5,
):
    test_data_text = get_caption_prompt(test_data['caption'])
    test_data_image = Image.open(test_data['image']).convert('RGB')
    test_data_id = test_data['idx']

    candidateidx2data = {
        data['idx']: {
            'caption': get_caption_prompt(data['caption']),
            'image': data['image'],
            'idx': data['idx'],
        }
        for data in candidate_set
    }
    test_data_id_list = [[test_data_id]]

    for _ in range(few_shot_num):
        new_test_data_id_list = []
        new_test_score_list = []
        for test_data_id_seq in test_data_id_list:
            # 避免添加重复的结果 将已经添加的进行过滤
            if len(test_data_id_list) >= 2:
                filtered_candidateidx2data = candidateidx2data.copy()
                filter_id_list = test_data_id_seq[:-1]
                for i in filter_id_list:
                    filtered_candidateidx2data.pop(i)
            else:
                filtered_candidateidx2data = candidateidx2data.copy()

            # 构建已经选好的ice + 测试样本的输入
            ice_id_seq = test_data_id_seq[:-1]
            lang_x = [candidateidx2data[idx]['caption'] for idx in ice_id_seq] + [
                test_data_text
            ]
            image_x = [
                Image.open(candidateidx2data[idx]['image']).convert('RGB')
                for idx in ice_id_seq
            ] + [test_data_image]

            # 对于每一个候选计算InfoScore
            info_score_list = []
            filtered_idx_list = list(filtered_candidateidx2data.keys())
            for batch in more_itertools.chunked(filtered_idx_list, batch_size):
                batch_data = [filtered_candidateidx2data[i] for i in batch]
                new_ice_lang_x = [data['caption'] for data in batch_data]
                new_ice_image_x = [
                    Image.open(data['image']).convert('RGB') for data in batch_data
                ]

                sub_info_score = get_info_score(
                    model,
                    tokenizer,
                    image_processor,
                    device=device,
                    ice_join_char='',
                    lang_x=lang_x,
                    new_ice_lang_x=new_ice_lang_x,
                    image_x=image_x,
                    new_ice_image_x=new_ice_image_x,
                    autocast_context=autocast_context,
                )
                info_score_list.extend(sub_info_score.cpu().numpy().tolist())

            score_array = torch.tensor(info_score_list)
            # 选出最高的InfoScore
            scores, indices = score_array.topk(beam_size)
            indices = indices.tolist()
            indices = list(
                map(
                    lambda x: filtered_candidateidx2data[filtered_idx_list[x]]['idx'],
                    indices,
                )
            )
            scores = scores.tolist()

            for idx, score in zip(indices, scores):
                new_test_data_id_list.append([idx, *test_data_id_seq])
                new_test_score_list.append(score)

        new_test_score_list, new_test_data_id_list = beam_filter(
            new_test_score_list, new_test_data_id_list, beam_size
        )
        test_data_id_list = new_test_data_id_list
    return {
        test_data_id: {'id_list': test_data_id_list, 'score_list': new_test_score_list}
    }


def gen_data(
    rank,
    lang_encoder_path,
    tokenizer_path,
    flamingo_checkpoint_path,
    cross_attn_every_n_layers,
    hf_root,
    precision,
    sample_data,
    train_dataset,
    sim_candidate_set_idx,
    save_path,
    gpu_ids,
):
    world_size = len(gpu_ids)
    process_device = f'cuda:{gpu_ids[rank]}'

    subset_size = len(sample_data) // world_size
    subset_start = rank * subset_size
    subset_end = (
        subset_start + subset_size if rank != world_size - 1 else len(sample_data)
    )
    subset = [sample_data[i] for i in range(subset_start, subset_end)]
    sub_sim_set_idx = sim_candidate_set_idx[subset_start:subset_end]
    sleep(180 * rank)
    model, image_processor, tokenizer, autocast_context = init_flamingo(
        lang_encoder_path,
        tokenizer_path,
        flamingo_checkpoint_path,
        cross_attn_every_n_layers,
        hf_root,
        precision,
        process_device,
    )
    tokenizer.pad_token = tokenizer.eos_token

    final_res = {}
    sub_res_basename = (
        os.path.basename(save_path).split('.')[0]
        + f'_rank:{rank}_({subset_start}, {subset_end}).json'
    )
    save_path = save_path.replace(os.path.basename(save_path), sub_res_basename)

    for i, test_data in enumerate(tqdm(subset)):
        candidate_set = [train_dataset[int(idx)] for idx in sub_sim_set_idx[i]]
        res = generate_single_sample_ice(
            model,
            tokenizer,
            image_processor,
            test_data,
            candidate_set,
            device=process_device,
            few_shot_num=args.few_shot_num,
            autocast_context=autocast_context,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
        )
        final_res.update(res)
        with open(save_path, 'w') as f:
            json.dump(final_res, f)
    return final_res


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    cache_dir = os.path.join(args.result_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    save_dir = os.path.join(args.result_dir, 'generated_data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sub_proc_save_dir = os.path.join(save_dir, 'sub_proc_data')
    if not os.path.exists(sub_proc_save_dir):
        os.makedirs(sub_proc_save_dir)

    save_file_name = (
        f'{args.model_name}-coco-{args.sim_method}-'
        f'beam_size:{args.beam_size}-few_shot:{args.few_shot_num}-'
        f'query_top_k:{args.query_top_k}_by_train_ds.json'
    )
    sub_save_path = os.path.join(sub_proc_save_dir, save_file_name)
    save_path = os.path.join(save_dir, save_file_name)

    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    # 加载数据集
    train_dataset = CocoDataset(
        args.train_coco_dataset_root, args.train_coco_annotation_file
    )
    # 使用启发式获取小集合
    if args.sim_method == 'caption':
        encoding_method = encode_text
    elif args.sim_method == 'image':
        encoding_method = encode_image
    else:
        raise ValueError('the sim_method error')

    # pre-calculate the cache feature for knn search
    sim_model_name = args.sim_model_type.split('/')[-1]
    train_cache_path = os.path.join(
        cache_dir, f'train-coco-{args.sim_method}-{sim_model_name}-feature.pth'
    )
    train_feature = load_feature_cache(
        args, train_cache_path, encoding_method, train_dataset, args.sim_method
    )
    _, test_sim_query_set_idx = recall_sim_feature(
        train_feature, train_feature, top_k=args.query_top_k + 1
    )

    test_sim_query_set_idx = test_sim_query_set_idx[:, 1:]

    sample_index = random.sample(
        [i for i in range(0, len(train_dataset))], args.sample_num
    )
    sample_data = [train_dataset[i] for i in sample_index]
    result = spawn(
        gen_data,
        args=(
            args.lang_encoder_path,
            args.tokenizer_path,
            args.flamingo_checkpoint_path,
            args.cross_attn_every_n_layers,
            args.hf_root,
            args.precision,
            sample_data,
            train_dataset,
            test_sim_query_set_idx,
            sub_save_path,
            args,
        ),
        nprocs=len(args.gpu_ids),
        join=True,
    )

    final_res_dict = dict()
    for res in result:
        final_res_dict.update(res)
    with open(save_path, 'w') as f:
        json.dump(final_res_dict, f)
