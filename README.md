# ICD-LM
ICD-LM: Configuring Vision-Language In-Context Demonstrations by
Language Modeling

## Prepare
```
git clone 
conda create -n iclm python=3.10
conda activate iclm
pip install -r requirements.txt

# install the openicl package
mkdir requirements_repo
cd requirements_repo
# for anonymous submit, it will fix in Formal version
git clone https://github.com/xxxx/OpenICL.git
cd OpenICL
git checkout -b coco_caption origin/coco_caption
pip install -e ./
cd ../..
```

## .env config
You should set the Environment varibles for dataset path and openflamingo path:
```
CHECKPOINT_PATH="./openflamingo"  # the checkpoint path you want to save
COCO_PATH="/path/to/mscoco"
VQAV2_PATH="/path/to/vqav2"
RESULT_DIR="/path/to/result"  # the dir to save result(checkpoint, inference metric, cache...)
```
The flamingo checkpoint path will download automatically.


## Datasets
### MSCOCO
We use the mscoco train dataset to generate. 
So you should prepare the mscoco2017 or mscoco2014

```
|-- mscoco
|    |
|    |- mscoco2017
|    |   |
|    |   |- train2017
|    |   |- val2017
|    |   |- annotations
|    |       |
|    |       |- captions_train2017.json
|    |       |- captions_val2017.json
|    |- mscoco2014
|        |
|        |- train2014
|        |- val2014
|        |- annotations
|            |
|            |- captions_train2014.json
|            |- captions_val2014.json
```

### VQAV2
We use the VQAV2 train dataset to generate the good ICD Sequences.
So you should prepare the VQAV2 dataset or if you can download datasets from huggingface you can use `configs/dataset/vqav2_online.yaml`. 
```bash
# For download the vqav2 dataset:
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -O /path/to/vqav2/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -O /path/to/vqav2/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -O /path/to/vqav2/
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -O /path/to/vqav2/

cd /path/to/vqav2/
unzip v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip


# for preprepare the dataset.
python src/dataset_module/preprocess/vqav2_hf.py --root_path /path/to/vqav2/
```
Then, set the `VQAV2_PATH` environment variable in `.env`.


#### 1. Generation the train dataset

```
python generate_data.py candidate_set_num=128 beam_size=10 few_shot_num=4 bs=128 gpu_ids="[0,1,2,3,4,5,6,7]"

python generate_data.py candidate_set_num=128 beam_size=10 few_shot_num=4 bs=128 gpu_ids="[0,1,2,3,4,5,6,7]" random_sample_candidate_set=true

```

#### 2. Train the ICLM Mode
```
python train.py
```
Args:
- `train`: Options are `query_img_ice_idx`, `query_img_ice_img_text`, `query_img_ice_img`, `query_img_ice_text`, `query_img_text_ice_img_text`. 'img' after 'query' indicates the addition of image information to the query sample. 'text' after 'query' indicates the addition of text information to the query sample. The same applies to 'ice'.
- `dataset`: Defines the dataset for ICE. For caption tasks, you can choose either coco2017 or coco2014; for VQA tasks, choose between vqav2_local or vqav2_online. This parameter also includes the dataset path and other relevant information.
- `task`: Options are `vqa` or `caption`, configuring parameters related to prompt
- `data_files`: Specifies the names of the JSON data files generated in the first step.
- `trainer_args`: the lightning triner args
- `lr`: learning rate
- `ex_name`: Name of the current experiment, which is also the name of the folder for saving experimental results. 
- `seed`: Sets the seed for random number generation.

#### 3. Use Flamingo Inference
If test the iclm model, you should set:
- `train`: Options are `query_img_ice_idx`, `query_img_ice_img_text`, `query_img_ice_img`, `query_img_ice_text`, `query_img_text_ice_img_text`. 'img' after 'query' indicates the addition of image information to the query sample. 'text' after 'query' indicates the addition of text information to the query sample. The same applies to 'ice'.
- `iclm_path`: Path to the model checkpoint.
- `test_iclm`: Set to true.

Other args;
- `dataset`: Defines the dataset for ICE. For caption tasks, choose either coco2017 or coco2014; for VQA tasks, choose between vqav2_local or vqav2_online.
- `task`: Options are `vqa` or `caption`, configuring parameters related to prompt
- `flamingo`: Flamingo model version, options include `flamingo_3B`, `flamingo_9B`.
- `index_data_num`: Number of items in the ICD training set, -1 for all.
- `test_data_num`: Number of items in the test set, -1 for all.
- `other_save_field`: etadata keys to save. For caption, possible values are `['single_caption', 'captions', 'image_id']`，for VQA `['question', 'answer', 'answers', 'question_id', 'image_id']`
- `inference_bs`：Batch size for inference. For a 3090 with 24G of memory, a setting of 4 is feasible for 16 shots.


## Ablation
### Generate data
**for caption**
```sh
# for one gpu defalut [0]
bash ./scripts/ablation/generate_data.sh caption coco2017 
# for multi gpu (4 gpus):
bash ./scripts/ablation/generate_data.sh caption coco2017 "[0,1,2,3]"
```
**for vqav2**
```sh
# for one gpu defalut [0]
bash ./scripts/ablation/generate_data.sh vqa vqav2_local
# for multi gpu (4 gpus):
bash ./scripts/ablation/generate_data.sh vqa vqav2_local "[0,1,2,3]"
```

### Train UniGen-ICLM
**for caption**
```sh
# for one gpu defalut [0]
bash ./scripts/ablation/train_iclm.sh caption coco2017 
# for multi gpu (4 gpus):
bash ./scripts/ablation/generate_data.sh caption coco2017 4
```
**for vqav2**
```sh
# for one gpu defalut [0]
bash ./scripts/ablation/train_iclm.sh vqa vqav2_local
# for multi gpu (4 gpus):
bash ./scripts/ablation/train_iclm.sh vqa vqav2_local 4
```


### Inference...
```sh
To be continue...
```
