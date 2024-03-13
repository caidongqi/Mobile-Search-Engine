import subprocess
import torch
import numpy as np
import torch.nn.functional as F
# 遍历 audio_num_blocks 的值从 1 到 12 1,5 5,9  9,12    


for text_num_blocks in range(1,9):
    # 构建执行命令
   
    
    #command = f"python test_vgg.py {audio_num_blocks} "
    #python test_vgg.py --text_num_blocks 8 --device "cuda:0"
    command = f"python audio_search_engine.py --text_num_blocks {text_num_blocks} "
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # parameter=one_image.total_params
    if result.returncode == 0:
        
        print(f"done")
    else:
        print(f"Command '{command}' failed with error: {result.stderr}")
    


# with open(txt_file_path, "w") as txtfile:
#     for i in range(32):
#         row_data = " ".join([f"{similarities[i, j]:.4f}" for j in range(32)])
#         txtfile.write(f"Row {i+1}: {row_data}\n")

# print(f"Similarities matrix saved to {txt_file_path}")

    # try:
    #     subprocess.run(command, shell=True, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Command '{command}' failed with error: {e}")
    # importlib.reload(one_image)
    # value= one_image.embed
    # embeddings.append(value)
    # embeddings=torch.stack([embeddings, value])
    # embeddings=np.concatenate([embeddings,value])
    
