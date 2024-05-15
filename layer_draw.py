import numpy as np
import os

filenames = ['t24_a{0}.txt'.format(i) for i in range(1,13)]
files = [os.path.join('results/clotho/lora/text_nohead/R10',file) for file in filenames]
counts = np.array([])
counts = [np.loadtxt(file) for file in files]

acc_list = []

for i,file in enumerate(files):
    acc_list.append(np.sum(counts[i]==1)/len(counts[i]))
    np.savetxt(file, counts[i], fmt='%i')

print(acc_list)

# for file in files:
#     if len(counts) == 0:
#         counts = np.loadtxt(file)
#     else:
#         counts = np.concatenate(np.loadtxt(file))


# def sort_matrix_by_ones(matrix):
#     # 统计每列中元素1的个数
#     ones_count_per_column = np.sum(matrix, axis=0)

#     # 获取排序后的列索引
#     sorted_column_indices = np.argsort(ones_count_per_column)[::-1]

#     # 根据排序后的索引重新排列矩阵的列
#     sorted_matrix = matrix[:, sorted_column_indices]

#     return sorted_matrix
import numpy as np

def sort_matrix_by_ones(matrix):
    # 统计每列中元素1的个数
    ones_count_per_column = np.sum(matrix, axis=0)

    # 获取排序后的列索引
    sorted_column_indices = np.argsort(ones_count_per_column)[::-1]

    # 记录排序前每一列的原始索引
    original_column_indices = np.arange(matrix.shape[1])

    # 根据排序后的索引重新排列矩阵的列
    sorted_matrix = matrix[:, sorted_column_indices]

    return sorted_matrix, original_column_indices[sorted_column_indices]




print(filenames)
print(np.array(counts))
sorted_matrix ,indices= sort_matrix_by_ones(np.array(counts))
np.savetxt('./results/agnews-t-figure.txt', sorted_matrix, fmt='%i')
np.savetxt('./results/clotho/sort.txt', indices, fmt='%i')



# for i in range(21,32):
#     acc_list = []
#     # for j in range(21,i+1):
#     #     acc_list.append(-1)
#     for j in range(21,33):
#         acc_list.append(round(np.sum(np.logical_and(counts[i-21]==counts[j-21] , counts[i-21] == 1))/np.sum(counts[i-21]==1)*100,2))
#     print(acc_list)



import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# Settings
data_ids = len(np.loadtxt(files[0]))  # total number of data points
inference_methods = len(filenames)  # number of different inference methods

# Randomly generate data
np.random.seed(0)
# data = np.random.randint(0, 2, (inference_methods, data_ids))

# Plotting using imshow for efficiency
plt.figure(figsize=(12, 6))
plt.imshow(sorted_matrix, cmap='bwr', aspect='auto', interpolation='none')  # using blue-white-red colormap for contrast
import matplotlib.ticker as ticker

# Your plotting code goes here

# Define a custom formatter function
def custom_formatter(x, pos):
    labels = {0: 'Incorrect', 1: 'Correct'}
    return labels.get(x, '')

plt.colorbar(label='Inference Outcome', ticks=[0, 1],  format=ticker.FuncFormatter(custom_formatter))
plt.xlabel('Data ID')
plt.ylabel('Inference layer')
plt.title('Matrix Plot of Inference Outcomes')
plt.yticks(range(inference_methods), [f"layer {i+1}" for i in range(inference_methods)])

# plt.show()
plt.savefig('./results/clotho/12layer-lora.pdf')