#!/bin/bash
source ~/.bashrc
source /opt/app/anaconda3/bin/activate imagebind
cd /home/u2021010261/pc/Mobile-Search-Engine
python image_process.py  > logs/image/method3.log 2>&1
