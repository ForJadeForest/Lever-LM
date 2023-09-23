from typing import Dict, List

import more_itertools
import torch
from PIL import Image
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO


def compute_cider(result_dict, annotations_path, reduce_cider=True):
    # create coco object and coco_result object
    coco = COCO(annotations_path)
    coco_result = coco.loadRes(result_dict)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()
    if reduce_cider:
        return coco_eval.eval['CIDEr']
    else:
        return coco_eval.imgToEval


@torch.inference_mode()
def get_cider_score(
    model,
    tokenizer,
    image_processor,
    device: str,
    ice_join_char: str,
    lang_x: List[str],
    image_x: List[Image.Image],
    candidate_set: Dict,
    batch_size: int,
    train_ann_path: str,
    gen_kwargs: Dict,
    autocast_context,
):
    output_dict = {}
    image_x = [image_processor(image) for image in image_x]
    cand_idx = sorted(list(candidate_set.keys()))
    for batch in more_itertools.chunked(cand_idx, batch_size):
        batch_data = [candidate_set[i] for i in batch]
        new_ice_lang_x = [data['text_input'] for data in batch_data]
        new_ice_image_x = [data['image'] for data in batch_data]

        # 2.1 拼接文本输入
        total_ice_lang_x_input = [
            ice_join_char.join([ice_lang_x] + lang_x) for ice_lang_x in new_ice_lang_x
        ]
        total_ice_lang_x_input = tokenizer(
            total_ice_lang_x_input, return_tensors='pt', padding=True
        ).to(device=device)

        batch_total_vision_x = [
            torch.stack([image_processor(ice_image_x)] + image_x, dim=0)
            for ice_image_x in new_ice_image_x
        ]
        total_vision_x = torch.stack(batch_total_vision_x, dim=0)

        total_vision_x = total_vision_x.unsqueeze(2).to(
            device=device, non_blocking=True
        )
        with autocast_context:
            outputs = model.generate(
                vision_x=total_vision_x,
                lang_x=total_ice_lang_x_input['input_ids'],
                attention_mask=total_ice_lang_x_input['attention_mask'].bool(),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                **gen_kwargs,
            )
        outputs = outputs.tolist()
        prompt_len = int(total_ice_lang_x_input['attention_mask'].shape[1])

        generated = tokenizer.batch_decode(
            [output[prompt_len:] for output in outputs],
            skip_special_tokens=True,
        )
        for i, data in enumerate(batch_data):
            output_dict[data['idx']] = {}
            output_dict[data['idx']]['prediction'] = generated[i]
            output_dict[data['idx']]['image_id'] = data['image_id']

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
    cider_score_info = compute_cider(pred_coco, train_ann_path, reduce_cider=False)
    cider_score = []
    for idx in cand_idx:
        img_id = candidate_set[idx]['image_id']
        cider_score.append(cider_score_info[img_id]['CIDEr'])

    return torch.tensor(cider_score)


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Cider(), "CIDEr"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f" % (method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
