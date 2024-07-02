import pickle
import torch
import numpy as np

K = 1
ground_truth_layer_path = f"/home/u2021010261/share/pc/Mobile-Search-Engine/results/coco_lora/R{K}/layers_0.txt"
predicted_layer_path = "/home/u2021010261/share/pc/Mobile-Search-Engine/parameters/image/coco/layers/N=2_S=1_v1_true.pkl"
predicted_layer_path_new = "/home/u2021010261/share/pc/Mobile-Search-Engine/parameters/image/coco/layers/N=2_S=1_v1_true_new.pkl"
# ground_truth_layer = 
with open(ground_truth_layer_path, 'r') as file:
    ground_truth_layers = [int(line.strip()) for line in file]
    print(np.mean(ground_truth_layers))

with open(predicted_layer_path, 'rb') as f:
    predicted_layer = pickle.load(f)
    # print(predicted_layer)
    print(np.mean([a.cpu() for a in predicted_layer]))

with open(predicted_layer_path_new, 'wb') as f:
    pickle.dump(ground_truth_layers, f)
    ground_truth_layers = [a for a in ground_truth_layers]
    print(np.mean(ground_truth_layers))