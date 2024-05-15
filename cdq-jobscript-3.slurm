#!/bin/bash
source ~/.bashrc
source /opt/app/anaconda3/bin/activate imagebind
cd /home/u2021010261/pc/Mobile-Search-Engine
python clotho_search_3.py > cdq-logs/new-N-1-K-30-test-3.log 2>&1
