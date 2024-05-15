#!/bin/bash
source ~/.bashrc
source /opt/app/anaconda3/bin/activate imagebind
cd /home/u2021010261/pc/Mobile-Search-Engine
python clotho_mobile-search_engine_lora.py > cdq-logs/audio-lora-search.log 2>&1
