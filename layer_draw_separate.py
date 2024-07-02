import numpy as np
import os

filenames = ['v{0}_t24.txt'.format(i) for i in range(1, 33)]
files = [os.path.join('results/flickr8k_lora_val_nohead/R1', file) for file in filenames]
counts = [np.loadtxt(file) for file in files]

acc_list = []

for i, file in enumerate(files):
    acc_list.append(np.sum(counts[i] == 1) / len(counts[i]))
    print(len(file))
    np.savetxt(file, counts[i], fmt='%i')

def sort_matrix_by_ones(matrix):
    ones_count_per_column = np.sum(matrix, axis=0)
    sorted_column_indices = np.argsort(ones_count_per_column)[::-1]
    original_column_indices = np.arange(matrix.shape[1])
    sorted_matrix = matrix[:, sorted_column_indices]
    return sorted_matrix, original_column_indices[sorted_column_indices]

sorted_matrix, indices = sort_matrix_by_ones(np.array(counts))

# 保存排序后的矩阵和列索引
np.savetxt('./results/sorted_matrix.txt', sorted_matrix, fmt='%i')
np.savetxt('./results/sorted_indices.txt', indices, fmt='%i')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# 从文件读取数据
sorted_matrix = np.loadtxt('./results/sorted_matrix.txt', dtype=int)
indices = np.loadtxt('./results/sorted_indices.txt', dtype=int)

# Settings
inference_methods = sorted_matrix.shape[0]  # number of different inference methods

# Plotting using imshow for efficiency
plt.figure(figsize=(12, 6))
plt.imshow(sorted_matrix, cmap='bwr', aspect='auto', interpolation='none')  # using blue-white-red colormap for contrast

# Define a custom formatter function
def custom_formatter(x, pos):
    labels = {0: 'Incorrect', 1: 'Correct'}
    return labels.get(x, '')

plt.colorbar(label='Inference Outcome', ticks=[0, 1], format=ticker.FuncFormatter(custom_formatter))
plt.xlabel('Data ID')
plt.ylabel('Inference layer')
plt.title('Matrix Plot of Inference Outcomes')
plt.yticks(range(inference_methods), [f"layer {i+1}" for i in range(inference_methods)])

# 保存图像
plt.savefig('./results/flickr8k_lora_val_nohead/32layer-lora.pdf')