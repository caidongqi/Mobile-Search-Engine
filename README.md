# Always-on Mobile Multimodal Embedding


## Table of Contents

- [Environment Installation](#environment-installation)
    - [ImageBind](#imagebind)
    - [ImageBind Trainer with LoRA fine-tuning](#imagebind-trainer-with-lora-fine-tuning)
    - [Code structure](#code-structure)
- [Usage](#usage)
    - [Search](#search)
    - [Fine-tuning](#fine-tuning)
    - [Inference](#inference)
    - [train_lumen_imagenet.py](#train_lumen_imagenet.py)




## Environment Installation
git clone 
### ImageBind
Install pytorch 1.13+ and other 3rd party dependencies.

```shell
conda create --name imagebind python=3.8 -y
conda activate imagebind

pip install -r requirements.txt
```

Install `matplotlib` when using the `train.py` script without the `--headless` argument.

Please follow the original [instructions](https://github.com/facebookresearch/ImageBind) for further information.

### Code structure
Put all plot scripts into `plot_scripts/`.
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

## Usage
### Search

In `search.py`, you can find an example of how to use the model for search images of target label. To try the `LoRA` fine-tuned model, change `lora=True`, set the fine-tuned model's path `lora_dir` and the parameters in `LoRA.apply_lora_modality_trunks()` within the script. To try the original ImageBind model, set `lora=False`.

**example explanation**: The `Imagenet` dataset contains 1000 classes, search the corresponding images for the three words(`stingray`,`cock`,`hen`) and obtain the precision and recall of the images we searched for.



### Fine-tuning

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

### Inference

In `test_imagent.py`, you can find an example of how to use the model for inference on ImageNet Dataset. To try the `LoRA` fine-tuned model, change `lora=True`, set the fine-tuned model's path `lora_dir` and the parameters in `LoRA.apply_lora_modality_trunks()` within the script. To try the original ImageBind model, set `lora=False`. And you can set the trunk blocks of each modlity when use `imagebind_model.imagebind_huge()`.



### train_lumen_imagenet.py
used to train lumen model

use the codes below to train experiment6:
 self.model = lumen6_model.imagebind_huge(pretrained=True,vision_num_blocks_1=1,vision_num_blocks_2=30,text_num_blocks=24)

use the codes below to train experiment1:
 self.model = lumen_model.imagebind_huge(pretrained=True,vision_num_blocks_1=1,vision_num_blocks_2=30,text_num_blocks=24)
      
other parameters(most can be changed in the function "parser"):
lora_dir:
parser.add_argument("--lora_checkpoint_dir", type=str, default="./.checkpoints/lora/lume_lora-666!!!")
device:
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")
....
trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
                      devices=[1,2,3], deterministic=True)