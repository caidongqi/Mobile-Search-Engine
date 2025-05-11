# Always-on Mobile Multimodal Embedding


## Table of Contents

- [Environment Installation](#environment-installation)

- [Code structure](#code-structure)
- [Usage](#usage)
    - [Search Demo](#search-demo)
    - [Fine-tuning](#fine-tuning)
    - [End-to-end System Workflow](#end-to-end-experiments-instruction)
    <!-- - [train_lumen_imagenet.py](#train_lumen_imagenet.py) -->




## Environment Installation [Typical install time: 10 mins]
Make sure to clone this repository recursively to include the submodules:

```bash
git clone --recurse-submodules -j8 https://github.com/fabawi/ImageBind-LoRA.git
```

For installation, please follow the original [usage instructions](#Usage).
Install `matplotlib` when using the `train.py` script without the `--headless` argument.

**Warning**: If you receive the following error -> "'FastAPI' object has no attribute 'debug'", upgrade `fastapi` to the latest version:

```bash
pip install --upgrade fastapi
```

Install pytorch 1.13+ and other 3rd party dependencies.

```shell
conda create --name imagebind python=3.8 -y
conda activate imagebind

pip install -r requirements.txt
```

Install `matplotlib` when using the `train.py` script without the `--headless` argument.

Please follow the original [instructions](https://github.com/facebookresearch/ImageBind) for further information.

## Code structure
TODO: Put all plot scripts into `plot_scripts/`.
```
├── api/
│   └── *data pre-processors*
├── datasets/
│   └── *data loader*
├── ImageBind-LoRA/
│   └── *lora training code*
├── lightning_logs/
│   └── *automated generated logs*
├── logs/
│   └── *mannually generated logs*
├── metrics/
│   └── *evaluation metrics and benchmarking scripts*
├── models/
│   └── *definition of neural network models and execution workflow*
├── plot_scripts/
│   └── *draw figures* 
├── audio_process.py
├── data.py
├── lumen_2_infer.py
├── lumen_imagenet.py
├── ...
└── train.py
```


# Usage


## Search demo

In `search.py`, you can find an example of how to use the model for search images of target label. To try the `LoRA` fine-tuned model, change `lora=True`, set the fine-tuned model's path `lora_dir` and the parameters in `LoRA.apply_lora_modality_trunks()` within the script. To try the original ImageBind model, set `lora=False`.
And you can set the trunk blocks of each modlity when use imagebind_model.imagebind_huge().

**example explanation**: The `Imagenet` dataset contains 1000 classes, search the corresponding images for the three words(`stingray`,`cock`,`hen`) and obtain the precision and recall of the images we searched for.



## Fine-tuning

Modify `train.py` to adapt to the training ImageNet data set, and the modified code is stored in `train_iamgenet.py`.

Below is the information about `train.py`.
To train the model, run:

```bash
python train.py --batch_size 12 --max_epochs 500 \
        --lora --lora_modality_names vision text \
        --self_contrast --datasets dreambooth
```

You can enable logging using `comet`, `wandb` or `tensorboard` by setting the `--loggers` argument to the chosen logger/s.
Make sure to install the respective logging packages beforehand as well as the necessary environment variables.

To specify the layers or modalities to apply LoRA to, 
use the `--lora_layer_idxs` and `--lora_modality_names` arguments. 
To override specific layer counts for a certain modality, you could target the modality specifically, 
e.g., add the following argument to specify LoRA for the first 6 layers of the vision trunk only:

```bash
--lora_layer_idxs_vision 1 2 3 4 5 6
```

To train on GPU (currently runs on a single GPU, but multi-GPU training will be added soon), set the `--device` argument:

```bash
--device cuda:0
```

The LoRA models used in `example.py` 
(checkpoints found in `.checkpoints/lora/550_epochs/` with postix `_dreambooth_last.safetensors`), 
was trained for ~2 hours on a 3080Ti with 12 GB VRAM, consuming 5.66 GB VRAM and ~4 GB RAM. The model converged to a similar state in less than 30 mins.

INFO:

8.0 M     **Trainable params**

1.2 B     **Non-trainable params**

1.2 B     **Total params**

4,815.707 **Total estimated model params size (MB)**


We set the train arguments as follows:

```bash

# installed comet-ml:
#       pip install comet-ml
# and set the env variables:
#       export COMET_API_KEY=<MY_API_KEY>
#       export COMET_WORKSPACE=<MY_WORKSPACE_NAME>
#       export COMET_PROJECT_NAME=Imagebind-lora

python train.py --batch_size 12 --max_epochs 550 --num_workers 4 \
                --lora --lora_modality_names vision text \
                --self_contrast --datasets dreambooth \
                --device cuda:0 --headless --loggers comet
```

**Note**: To perform linear probing (optimizing the last layer of each modality's head only), maintain all arguments, 
replacing `--lora` with `--linear_probing` (Both cannot be set in the same run). 
On running `--lora` in the next training session/s, the checkpoint of the heads is automatically loaded and saved,
assuming the `--lora_checkpoint_dir` remains the same.


## End-to-end experiments instruction:

## E2E (end to end) [Expected run time: 1 hour with provided check points]
To construct an e2e system, you have 4 steps to go. In `run_dataset.py`, you can see the whole pipeline of clotho dataset,  the same pattern applies when using other datasets.

### Step 1: Get every embedding of the data with different model layers.
#### In our technique, we need to embed every data dynamically, so we need to prepare the embeddings of different model layers.
In `get_embedding_cltho.py`, we can compute the embeddings of clotho dataset at a specific audio layer, so at step 1 in `run_dataset.py`, you need to iterate all the audio layers (form 1 to the whole audio layer of Imagebind).

Parameters of `get_embedding_cltho.py`:
Input: 
--lora_layers 'defines the layer of the model'
--lora dir 'the path of lora parameters'
--embedding_dir 'the path to save the embeddings'
--dataset 'the name to the dataset'

Output:

Relevant dataset embedings saved in `embedding_dir`.

### Step 2: Inference the dataset at different model layers to get the data prediction results.
#### In this step, our goal is to get the prediction result of every single data in dataset, for example, at audio layer=7, the output result for a silgle data's R@N is 0, while at layer=9, it might be 1.
In `test_clotho_val.py`, we compute all the predictions of a dataset at a certain model layer, the results are 0/1, '0' means false while '1' means correct. At step 2, we iterate all the layers from 1-full layers, then we get the whole results of every data at different layers.

Parameters of `test_clotho_val.py`:
Input: 
--audio_num_blocks   'defines the layer of the model'
--lora dir   'the path of lora parameters'
--embedding_path  'the path to save the embeddings'
--version  'the tag of the experiment, often contains the name of the dataset and the method of the experiment (for example, lora or not, with lora head or not), it is used for recogizing the files'

Output:
every data predictions results saved in txt files, often at 'results/clotho_head/R{N}'

### Step3: Get the min layer of every data
Use the the txt files to get the min layer that the result is 1. In step 2, we only get the results of every model layer for each data, but the utimate goal is to get the least layer every data needs to be retrieved, therefore, we need to find out the first layer that the result is '1', and the python file`get_layers_clotho.py` can do this job.

Parameters of `get_layers_clotho.py`:

Input :

Txt files got in step 2.

Output:

the least layers every data needs saved in txt files, often at 'results/clotho_head/R{N}/layers.txt' 



### Step4: Use the labels got at step3 to train the predictor model

We feed the model with the embeddings of the dataset at a certain model layer N , and the least layer every data needs (the labels), to make the model predicts how many layers for a certain data needs.

Parameters of `model_predict_lora.py`:

Input:

Labels for each data, often at 'results/clotho_head/R{N}/layers.txt' 

Output:

model checkpoints

### Step5: Use the model got in Step4 to dynamically embed th dataset and get the prediction results to choose top `Q` results for fine-grained search.

Input:

N: how many layers you want to feed the models, usually, the bigger N ,the better accuracy, but also the longer calculation time 

S: it is the same `S` in R@S labels you feed to the model, the bigger S, the less layers model predicts , but the higher accuracy for predictor models 

Q: the top Q results got in dynamic search, the bigger Q, the more results we need to save ,but more accurate

Output:

the dynamic search accuracy\the e2e search accuracy


