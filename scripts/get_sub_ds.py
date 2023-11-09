import json

if __name__ == '__main__':
    f_n = './sup_result/generated_data/caption-coco2017-only_y_loss-OpenFlamingo-9B-vitl-mpt7b-random-beam_size:5-few_shot:2-candidate_set_num:64-sample_num:5000.json'
    
    data = json.load(open(f_n))
    
    split_num_list = [1000, 3000]
    
    for n in split_num_list:
        new_data = {
            k: v for i, (k, v) in enumerate(data.items())  if i < n
        }
        print(len(new_data))
        json.dump(new_data, open(f_n.replace('sample_num:5000', f'sample_num:{n}'), 'w'))