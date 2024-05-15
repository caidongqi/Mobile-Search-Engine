python imagenet_mobile-search_engine_lora_new.py --S 10 --split train --N 2 --K 10 > logs/cdq/large_A_S_10_K_10.log 2>&1 &

python imagenet_mobile-search_engine_lora_new.py --S 10 --split train --N 2 --K 20 --device cuda:1 > logs/cdq/large_A_S_10_K_20.log 2>&1 &

python imagenet_mobile-search_engine_lora_new.py --S 10 --split train --N 2 --K 40 --device cuda:1 > logs/cdq/large_A_S_10_K_40.log 2>&1 &

python imagenet_mobile-search_engine_lora_new.py --S 10 --split train --N 2 --K 60 --device cuda:1 > logs/cdq/large_A_S_10_K_60.log 2>&1 &