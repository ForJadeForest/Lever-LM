# Lever-LM
Lever LM: Configuring In-Context Sequence to Lever Large Vision Language Models

## Prepare
```
git clone 
conda create -n leverlm python=3.10
conda activate leverlm
pip install -r requirements.txt

# install the openicl package
mkdir requirements_repo
cd requirements_repo
# for anonymous submit, it will fix in Formal version
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
python open_mmicl/dataset_module/preprocess/vqav2_hf.py --root_path /path/to/vqav2/
```
Then, set the `VQAV2_PATH` environment variable in `.env`. (If you use `vaqv2_local` as dataset)


#### 1. Generation the train dataset

```shell
# for coco2017 image captioning
bash scripts/generate_data.sh caption coco2017 "[0,1,2,3]" 

# for vqav2
bash scripts/generate_data.sh vqa vqav2_local "[0,1,2,3]"
# We support vqav2 dataset of hf. It will download the dataset automatically.
bash scripts/generate_data.sh vqa vqav2_online "[0,1,2,3]"
```

#### 2. Train the Lever-LM Mode
```shell
# for coco2017 image captioning
bash scripts/train_lever_lm.sh caption coco2017 1 query_img_icd_img_text

# for vqav2
bash scripts/train_lever_lm.sh vqa vqav2_local 1 query_img_text_icd_img_text
# or use hf vqav2 dataset
bash scripts/train_lever_lm.sh vqa vqav2_online 1 query_img_text_icd_img_text
```

```
python train.py
```
Args:
- `train`: Options are `query_img_icd_idx`, `query_img_icd_img_text`, `query_img_icd_img`, `query_img_icd_text`, `query_img_text_icd_img_text`. 'img' after 'query' indicates the addition of image information to the query sample. 'text' after 'query' indicates the addition of text information to the query sample. The same applies to 'icd'.
- `dataset`: Defines the dataset for In-context Learning. For caption tasks, you can choose either coco2017 or coco2014; for VQA tasks, choose between vqav2_local or vqav2_online. This parameter also includes the dataset path and other relevant information.
- `task`: Options are `vqa` or `caption`, configuring parameters related to prompt
- `data_files`: Specifies the names of the JSON data files generated in the first step.
- `trainer_args`: the lightning triner args
- `lr`: learning rate
- `ex_name`: Name of the current experiment, which is also the name of the folder for saving experimental results. 
- `seed`: Sets the seed for random number generation.

#### 3. Use Flamingo Inference
```shell
# for coco2017 image captioning
bash scripts/inference.sh caption coco2017 0 query_img_icd_img_text

# for vqav2
bash scripts/inference.sh vqa vqav2_local 0 query_img_text_icd_img_text
# or use hf vqav2 dataset
bash scripts/inference.sh vqa vqav2_online 0 query_img_text_icd_img_text
# You can use a vqav2 sub-val set to validate the performance, which only contain 1w samples. 
bash scripts/inference.sh vqa vqav2_local_sub 0 query_img_text_icd_img_text
```

```shell
python inference_flamingo.py
```
If test the lever_lm model, you should set:
- `train`: Options are `query_img_icd_idx`, `query_img_icd_img_text`, `query_img_icd_img`, `query_img_icd_text`, `query_img_text_icd_img_text`. 'img' after 'query' indicates the addition of image information to the query sample. 'text' after 'query' indicates the addition of text information to the query sample. The same applies to 'icd'.
- `lever_lm_path`: Path to the model checkpoint.
- `test_lever_lm`: Set to true.
- `random_order_lever_lm_iocd`: If set `True`, the icd configuration generated by Lever-LM will be randomly shuffled. 
- `default_cpk_key`: The checkpoint key word. You can set it to `last` or `min_loss`
- `ex_name`: Name of the current experiment, which is also the name of the folder for saving inference results. 


Other args;
- `dataset`: Defines the dataset for In-context Learning. For caption tasks, choose either coco2017 or coco2014; for VQA tasks, choose between vqav2_local or vqav2_online.
- `task`: Options are `vqa` or `caption`, configuring parameters related to prompt
- `flamingo`: Flamingo model version, options include `flamingo_3B`, `flamingo_9B`.
- `index_data_num`: Number of items in the ICD training set, -1 for all.
- `test_data_num`: Number of items in the test set, -1 for all.
- `inference_bs`：Batch size for inference. For a 3090 with 24G of memory, a setting of 4 is feasible for 16 shots.
- `shot_num_list`: The shot num you want to test.

Test Retrieval-based Method: 
- `test_random`: Use RS as the retrieval method.
- `test_t2t`: Use STTR as the retrieval method.
- `test_i2t`: Use SITR as the retrieval method.
- `test_i2i`: Use SIIR as the retrieval method.
- `mmtopk_clip_name`: CLIP model name to calculate the similarity.
- `mmtopk_reversed_order`: If set `True`, the rightmost ICD is the most similar, while set `False`, the leftmost ICD is the most similar.
