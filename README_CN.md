# Lever-LM
Lever LM: Configuring In-Context Sequence to Lever Large Vision Language Models

## 环境配置
```
git clone 
conda create -n leverlm python=3.10
conda activate leverlm
pip install -r requirements.txt
```

## .env 文件设置
You should set the Environment varibles for dataset path and openflamingo path:
```
CHECKPOINT_PATH="./openflamingo"  # the checkpoint path you want to save
COCO_PATH="/path/to/mscoco"
VQAV2_PATH="/path/to/vqav2"
RESULT_DIR="/path/to/result"  # the dir to save result(checkpoint, inference metric, cache...)
```
Flamingo checkpoint会自动下载在CHECKPOINT_PATH


## Datasets
### MSCOCO
在论文中，我们主要是用mscoco来进行Image Captioning实验，所以你需要准备mscoco2017或者mscoco2014。我们推荐使用mscoco2017。

coco数据集文件夹设置
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
我们使用VQAv2 train dataset来在VQA上进行实验。
1. 如果你使用`vqav2_local.yaml`，你需要手动下载数据集：

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
下载完之后，在`.env`中设置`VQAV2_PATH`环境变量。

2. 或者你可以直接从huggingface上自动进行下载(需要使用`configs/dataset/vqav2_online.yaml`配置)



#### 1. 构建Lever-LM训练数据集

```shell
# for coco2017 image captioning
bash scripts/generate_data.sh caption coco2017 "[0,1,2,3]" 

# for vqav2
bash scripts/generate_data.sh vqa vqav2_local "[0,1,2,3]"
# We support vqav2 dataset of hf. It will download the dataset automatically.
bash scripts/generate_data.sh vqa vqav2_online "[0,1,2,3]"
```

##### `generate_data.py` 参数设置
对应参数文件位于：`configs/inference.yaml`
1. `infer_model`: infer_model的模型种类，目前支持Open falamingo和IDEFICS-9B。具体参数细节参考xxx
2. `dataset`: 使用的数据集，可选值：`mscoco2017`, `mscoco2014`, `vqav2_local`, `vqav2_online`。具体参数细节参考xxx
3. `task`: 具体的任务种类，目前可选值`caption`和`vqa`。具体参数细节参考xxx
4. `sampler`: 获取sub-supporting set的sampler方法，目前支持`img_sim_sampler`, `text_sim_sampler`, `rand_sampler`。具体参数细节参考xxx
5. `beam_size`: 生成数据使用的beam size。
6. `few_shot_num`: 需要构建多长的few-shot数据
7. `batch_size`: Scorer计算的batch_size。不要超过sampler.candidate_num。
8. `device`: `cuda`，目前不支持`cpu`
9. `precision`: 模型计算的精度。
10. `sampler_num`: anchor_set 的数量大小。
11. `construct_order`: 构建ICD sequences的顺序，可选值为`left`, `right`。如果为`left`：[ICD1, Query] -> [ICD1, ICD2, Query]。`right`：[ICD1, Query] -> [ICD2, ICD1, Query]
12. `scorer`: 构建数据集所使用的scorer。可选值: `infoscore`, `cider`。
13. `gpu_ids`: 需要使用的GPU序号。值为一个列表。
14. `sleep_time`: 在初始化时同时加载几个模型到GPU会需要大量的内存，因此为了缓解内存压力，我们会每隔`sleep_time`时间开始初始化下一个GPU的模型。



#### 2. Train the Lever-LM Mode
```shell
# for coco2017 image captioning
bash scripts/train_lever_lm.sh caption coco2017 1 query_img_icd_img_text

# for vqav2
bash scripts/train_lever_lm.sh vqa vqav2_local 1 query_img_text_icd_img_text
# or use hf vqav2 dataset
bash scripts/train_lever_lm.sh vqa vqav2_online 1 query_img_text_icd_img_text
```

##### `train.py` 参数设置
- `train`：选项包括`query_img_icd_idx`、`query_img_icd_img_text`、`query_img_icd_img`、`query_img_icd_text`、`query_img_text_icd_img_text`。在'query'之后的'img'表示在查询样本中添加图像信息。在'query'之后的'text'表示在查询样本中添加文本信息。'icd'同样适用于这个规则。
- `dataset`: 使用的数据集，可选值：`mscoco2017`, `mscoco2014`, `vqav2_local`, `vqav2_online`。具体参数细节参考xxx
- `task`: 具体的任务种类，目前可选值`caption`和`vqa`。具体参数细节参考xxx
- `data_files`：指定在第一步生成的JSON数据文件的名称。
- `trainer_args`：lightning训练器参数。具体参考[Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api)
- `lr`：学习率。
- `ex_name`：当前实验的名称，也是保存实验结果的文件夹名称。
- `seed`：设置随机数生成的种子。


#### 3. Inference
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

##### Inference 参数设置
如果要测试lever_lm模型，需要设置以下参数：
- `train`：可选项为`query_img_icd_idx`、`query_img_icd_img_text`、`query_img_icd_img`、`query_img_icd_text`、`query_img_text_icd_img_text`。在"query"之后加上"img"表示将图像信息添加到查询样本中，加上"text"表示将文本信息添加到查询样本中。对于"icd"也是同样的规则。
- `lever_lm_path`：模型checkpoint的路径。如果设置为null，则会在`model_cpk/task_name/ex_name`中寻找。
- `default_cpk_key`: 当`lever_lm_path`为null的时候才有效。可选值为`last`, `min_vl`, `min_tl`。
    - `last` 会自动在`model_cpk/task_name/ex_name`搜寻最后一个epoch的checkpoint。
    - `min_vl`: 会自动在`model_cpk/task_name/ex_name`搜寻最小val loss版本。
    - `min_tl`: 会自动在`model_cpk/task_name/ex_name`搜寻最小train loss版本。
- `test_lever_lm`：设置为true。
- `random_order_lever_lm_iocd`：如果设置为`True`，Lever-LM生成的icd配置将被随机打乱。(暂时弃置)
- `default_cpk_key`：检查点关键字。可以设置为`last`或`min_loss`。
- `ex_name`：当前实验的名称，也是保存推理结果的文件夹名称。

其他参数：
- `dataset`: 使用的数据集，可选值：`mscoco2017`, `mscoco2014`, `vqav2_local`, `vqav2_online`。具体参数细节参考xxx
- `task`：具体的任务种类，目前可选值`caption`和`vqa`。具体参数细节参考xxx
- `infer_model`: infer_model的模型种类，目前支持Open falamingo和IDEFICS-9B。具体参数细节参考xxx
- `index_data_num`：ICD训练集中的项目数量，-1表示全部。
- `test_data_num`：测试集中的项目数量，-1表示全部。
- `inference_bs`：推理的批处理大小。对于具有24G内存的3090显卡，设置为4适用于16shot。
- `shot_num_list`：想要测试的样本数量。

测试基于检索的方法：
- `test_random`：使用随机检索（RS）方法。
- `test_t2t`：使用STTR作为检索方法。
- `test_i2t`：使用SITR作为检索方法。
- `test_i2i`：使用SIIR作为检索方法。
- `mmtopk_clip_name`：用于计算相似度的CLIP模型名称。
- `mmtopk_reversed_order`：如果设置为`True`，最右边的ICD是最相似的；如果设置为`False`，最左边的ICD是最相似的。
