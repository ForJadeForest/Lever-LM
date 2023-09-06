# ICLM
ICLM: Auto Generation In-Context Example with Language Model

## Environment
```
conda create -n name python=3.10
pip install -r requirements.txt
```

## .env config
You should set the Environment varibles for dataset path and openflamingo path:
```
CHECKPOINT_PATH="./openflamingo"
COCO_PATH="/path/to/mscoco"
```
The flamingo checkpoint path will download automatically.


## Datasets
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

