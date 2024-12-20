## Installation
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

## Search

In `search.py`, you can find an example of how to use the model for search images of target label. To try the `LoRA` fine-tuned model, change `lora=True`, set the fine-tuned model's path `lora_dir` and the parameters in `LoRA.apply_lora_modality_trunks()` within the script. To try the original ImageBind model, set `lora=False`.

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

## Inference

In `test_imagent.py`, you can find an example of how to use the model for inference on ImageNet Dataset. To try the `LoRA` fine-tuned model, change `lora=True`, set the fine-tuned model's path `lora_dir` and the parameters in `LoRA.apply_lora_modality_trunks()` within the script. To try the original ImageBind model, set `lora=False`. And you can set the trunk blocks of each modlity when use `imagebind_model.imagebind_huge()`.


## E2E 
To construct an e2e system, you have 4 steps to go. In `run_dataset.py`, you can see the whole pipeline of clotho dataset,  the same pattern applies when using other datasets.

### Step 1: Get every embedding of the data with different model layers.
#### In our technique, we need to embed every data dynamically, so we need to prepare the embeddings of different model layers.
In `get_embedding_cltho.py`, we can compute the embeddings of clotho dataset at a specific audio layer, so at step 1 in `run_dataset.py`, you need to iterate all the audio layers(form 1 to the whole audio layer of Imagebind).

Parameters of `get_embedding_cltho.py`:
Input: 
--lora_layers 'defines the layer of the model'
--lora dir 'the path of lora parameters'
--embedding_dir 'the path to save the embeddings'
--dataset 'the name to the dataset'

### Step 2: Inference the dataset at different model layers to get the data prediction results.
#### In this step, our goal is to get the prediction result of every single data in dataset, for example, at audio layer=7, the output result for a silgle data's R@N is 0, while at layer=9, it might be 1.
In `test_clotho_val.py`, we compute all the predictions of a dataset at a certain model layer, the results are 0/1, '0' means false while '1' means correct. At step 2, we iterate all the layers from 1-full, then we get the whole results of every data at different layers.

Parameters of `test_clotho_val.py`:
Input: 
--audio_num_blocks 'defines the layer of the model'
--lora dir 'the path of lora parameters'
--embedding_path 'the path to save the embeddings'
--version 'the tag of the experiment, often contains the name of the dataset and the method of the experiment (lora or not, with lora head or not)'

Output:
txt files, often at 'results/clotho_head/R{N}'

### Step3: Get the min layer of every data
Use the the txt files to get the min layer that the result is 1.

### Step4: Use the labels got in Step3 to dynamically embed th dataset and get the prediction results.
