# ICLM
ICLM: Auto Generation In-Context Example with Language Model

## Prepare
```
git clone 
conda create -n iclm python=3.10
conda activate iclm
pip install -r requirements.txt

# install the openicl package
mkdir requirements_repo
cd requirements_repo
git clone https://github.com/ForJadeForest/OpenICL.git
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
参数 (Args):
- `train`: 可选值为 `clip_ice_text_image`, `clip_ice_text`, `idx_base`，代表三种ICLM模型。
- `dataset`: 定义用于ICE的数据集。对于caption任务，可以选择coco2017或coco2014；对于vqa任务，可以选择vqav2_local或vqav2_online。此参数还包括数据集的路径和其他相关信息。
- `task`: 可选值为 `vqa` 或 `caption`，配置构建prompt的相关参数。
- `data_files`: 指定第一步生成的json数据文件名。
- `epochs`: 定义训练的周期数。
- `train_ratio`: 从生成的数据集中选择`train_ratio`的比例作为ICLM模型的训练数据，其余作为验证数据。
- `lr`: 设置学习率。
- `batch_size`: 指定训练批次的大小。
- `num_workers`: 设置训练时dataloader的num_workers参数。
- `warm_up_ratio`: 定义进行warmup的步骤数。
- `val_step`: 设定多少步执行一次验证。
- `result_dir`: 结果保存路径，默认无需更改。
- `ex_name`: 当前实验的名称，同时也是保存实验结果的文件夹名。
- `device`: 指定训练使用的设备。
- `device_num`: 设置使用的GPU数量。
- `precision`: 定义训练时的数值精度。
- `seed`: 设置随机数生成的种子。


#### 3. Use Flamingo Inference
If test the iclm model, you should set:
- `train`: 可选值为 `clip_ice_text_image`, `clip_ice_text`, `idx_base`，代表三种ICLM模型。
- `iclm_path`: 模型checkpoint路径
- `test_iclm`: 设置为true

Other args;
- `dataset`: 定义用于ICE的数据集。对于caption任务，可以选择coco2017或coco2014；对于vqa任务，可以选择vqav2_local或vqav2_online。此参数还包括数据集的路径和其他相关信息。
- `task`: 可选值为 `vqa` 或 `caption`，配置构建prompt的相关参数。
- `flamingo`: `flamingo` 模型版本，可选值`flamingo_3B`, `flamingo_9B`
- `index_data_num`: 用于构建ICE的train set的数量，-1表示全部
- `test_data_num`: 测试集的数量，-1表示全部
- `other_save_field`: 需要保存的一些metainfo key。对于caption，可以为`['single_caption', 'captions', 'image_id']`，对于vqa，可以为`['question', 'answer', 'answers', 'question_id', 'image_id']`
- `inference_bs`：推理的batc_hsize 对于3090 24G显存设置为4在16shot可行


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
