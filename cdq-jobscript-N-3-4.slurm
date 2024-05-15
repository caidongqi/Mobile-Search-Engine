#!/bin/bash
source ~/.bashrc
source /opt/app/anaconda3/bin/activate imagebind
cd /home/u2021010261/pc/Mobile-Search-Engine
python clotho_search_4.py > cdq-logs/N-3-test-4.log 2>&1
