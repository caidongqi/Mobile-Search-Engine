#!/bin/bash
source ~/.bashrc
source /opt/app/anaconda3/bin/activate imagebind
cd /home/u2021010261/pc/Mobile-Search-Engine
python imagenet_mobile-search_engine_lora_new.py --S 60 --split train --N 2 --K 100 > logs/cdq/large_A_S_60.log 2>&1
