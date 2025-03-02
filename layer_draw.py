import numpy as np
import os

filenames = ['v{0}_t24.txt'.format(i) for i in range(1,33)]
files = [os.path.join('results/flickr8k_lora_val_nohead/R1',file) for file in filenames]
counts = np.array([])
counts = [np.loadtxt(file) for file in files]

acc_list = []

for i,file in enumerate(files):
    acc_list.append(np.sum(counts[i]==1)/len(counts[i]))
    print(len(file))
    np.savetxt(file, counts[i], fmt='%i')

#print(counts)

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

def merge_columns(matrix):
    #matrix = np.array(matrix)  # 将输入的矩阵转换为 NumPy 数组
    rows, cols = matrix.shape
    assert cols % 5 == 0, "矩阵的列数必须是5的倍数"
    
    # 创建合并后的矩阵
    merged_cols = cols // 5
    merged_matrix = np.zeros((rows, merged_cols), dtype=int)
    
    for i in range(merged_cols):
        # 取出每组的五列
        group = matrix[:, i*5:(i+1)*5]
        # 计算每列的1的个数
        ones_count = np.sum(group, axis=0)
        # 选取1最多的列的索引
        max_ones_index = np.argmax(ones_count)
        # 将1最多的列赋值到合并后的矩阵中
        merged_matrix[:, i] = group[:, max_ones_index]
    
    return merged_matrix


#sorted_counts=merge_columns(np.array(counts))
sorted_matrix ,indices= sort_matrix_by_ones(np.array(counts))
np.savetxt('./results/agnews-t-figure.txt', sorted_matrix, fmt='%i')
np.savetxt('./results/coco_lora_val/sort.txt', indices, fmt='%i')



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
plt.savefig('./results/flickr8k_lora_val_nohead/32layer-lora.pdf')