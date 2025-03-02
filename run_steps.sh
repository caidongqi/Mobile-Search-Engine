# 通过不同的lora ckpt来获得image的coarse grained embedding
python run_get_embeddings_coco.py

# ------------------------------------------------
# 运行单次imagebind推理 (baseline)
python run_infer_coco_lora.py

#得到每个图片对应的层数
python run_get_labels.py

#根据得到的label进行模型训练
python run_get_models_lora.py

# ------------------------------------------------
# 运行端到端实验
# 1. 运行生成所有需要运行的脚本名
python run_e2e_coco_command_lora_max.py
#python run_e2e_coco_command_lora_min.py

# 2. 运行所有脚本，测试系统在不同超参数下的效果
python run_e2e_coco_exe_lora_max.py
#python run_e2e_coco_exe_lora_min.py