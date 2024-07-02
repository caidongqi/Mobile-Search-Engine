import torch
import os
root=f"parameters/image/coco/train_10"
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for i in range(1,33):
    cur_embedding= torch.load( os.path.join(root, f'embeddings_{i}_lora.pth'),map_location=torch.device(device))['vision_embeddings'][:11840]
    print(len(cur_embedding))
    torch.save({
        'vision_embeddings': cur_embedding
    }, f'parameters/image/coco/train_10/embeddings_{i}_lora.pth')