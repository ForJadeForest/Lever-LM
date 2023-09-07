if __name__ == '__main__':
    import json
    from pathlib import Path
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path', type=str
    )
    args = parser.parse_args()
    root = Path(args.root_path)
    train_ques = root / 'v2_OpenEnded_mscoco_train2014_questions.json'
    train_ann = root / 'v2_mscoco_train2014_annotations.json'

    val_ques = root / 'v2_OpenEnded_mscoco_val2014_questions.json'
    val_ann = root / 'v2_mscoco_val2014_annotations.json'

    save_path = root / 'vqav2_hf'
    if not save_path.exists():
        save_path.mkdir()
        
    
    ques = json.load(open(train_ques))
    ann = json.load(open(train_ann))
    quesid2question = {}
    for q in ques['questions']:
        quesid2question[q['question_id']] = q['question']
    total_data = []
    for a in ann['annotations']:
        a['question'] = quesid2question[a['question_id']]
        total_data.append(a)
    ann['annotations'] = total_data
    with open(save_path / 'vqav2_mscoco_train2014.json', 'w')as f:
        json.dump(ann, f)
        
    ques = json.load(open(val_ques))
    ann = json.load(open(val_ann))
    quesid2question = {}
    for q in ques['questions']:
        quesid2question[q['question_id']] = q['question']
    total_data = []
    for a in ann['annotations']:
        a['question'] = quesid2question[a['question_id']]
        total_data.append(a)
    ann['annotations'] = total_data
    with open(save_path / 'vqav2_mscoco_val2014.json', 'w')as f:
        json.dump(ann, f)
