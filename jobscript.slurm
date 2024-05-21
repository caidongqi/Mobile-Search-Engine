#!/bin/bash
source ~/.bashrc
source /opt/app/anaconda3/bin/activate imagebind
cd /home/u2021010261/share/pc/Mobile-Search-Engine
python imagenet_mobile-search_engine_lora_new.py --split 'val'   > logs/image/test_e2e_2.log 2>&1
python text_layers.py  > logs/image/getlabels_0518_13:05.log 2>&1