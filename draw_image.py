import numpy as np
import matplotlib.pyplot as plt

# Data grouped in sets of three rows, where each set is a distinct plot group
# data = [
#     [0.21052631578947367, 0.21052631578947367, None, None, None],
#     [0.40669856459330145, 0.40669856459330145, None, None, None],
#     [0.2966507177033493, 0.2966507177033493, None, None, None],
#     [0.8660287081339713, 0.8720379146919431, 0.8599033816425121, None, None],
#     [0.7440191387559809, 0.6919431279620853, 0.7971014492753623, None, None],
#     [0.8732057416267942, 0.8672985781990521, 0.8792270531400966, None, None],
#     [0.9441786283891547, 0.9095022624434389, 0.970873786, 0.9271844660194175, None],
#     [0.9298245614035088, 0.8687782805429864, 0.9611650485436893, 0.9368932038834952, None],
#     [0.9441786283891547, 0.9140271493212669, 0.9611650485436893, 0.9320388349514563, None],
#     [0.9700956937799043, 0.9181034482758621, 0.9905660377358491, 0.8867924528301887, 0.9433962264150944],
#     [0.9748803827751196, 0.9224137931034483, 0.9952830188679245, 0.8867924528301887, 0.9528301886792453],
#     [0.972488038, 0.9310344827586207, 0.9858490566037735, 0.8867924528301887, 0.9433962264150944]
# ]
with open('sort.csv', 'r') as file:
        lines = file.readlines()
num_layer=24
import csv

# 读取CSV文件并处理成指定格式的数据列表
def process_csv(file_path,num_layer):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for i,row in enumerate(reader):
            if i%2==1:
                # 提取数值部分
                values = [float(val) for val in row]
                # 根据层数添加None来对齐数据
                while len(values) < num_layer:
                    values.append(None)
                data.append(values)

    return data

# 测试
data = process_csv('sort.csv',num_layer)
print(data)

# data = [
#     [0.202,0.202, None, None, None],
#     [0.6497675665295803,0.6274703557312253,0.6720647773279352, None, None],
#     [0.191,0.1918997107039537,0.20721649484536084,0.17421953675730112, None],
#     [0.8660287081339713, 0.8720379146919431, 0.8599033816425121, None, None],
#     [0.7440191387559809, 0.6919431279620853, 0.7971014492753623, None, None],
#     [0.8732057416267942, 0.8672985781990521, 0.8792270531400966, None, None],
#     [0.9441786283891547, 0.9095022624434389, 0.970873786, 0.9271844660194175, None],
#     [0.9298245614035088, 0.8687782805429864, 0.9611650485436893, 0.9368932038834952, None],
#     [0.9441786283891547, 0.9140271493212669, 0.9611650485436893, 0.9320388349514563, None],
#     [0.9700956937799043, 0.9181034482758621, 0.9905660377358491, 0.8867924528301887, 0.9433962264150944],
#     [0.9748803827751196, 0.9224137931034483, 0.9952830188679245, 0.8867924528301887, 0.9528301886792453],
#     [0.972488038, 0.9310344827586207, 0.9858490566037735, 0.8867924528301887, 0.9433962264150944]
# ]

# Process data to calculate means and standard deviations for the first column, and prepare the rest for bar plots
processed_data = []
for i in range(0, len(data), 1):
    subset = data[i:i+1]
    means = [np.mean([x[0] for x in subset])]
    stds = [np.std([x[0] for x in subset], ddof=1)]
    
    # Max layers across subsets for alignment
    max_layers = max(len(row) for row in subset)
    for layer in range(1, max_layers):
        layer_values = [x[layer] for x in subset if x[layer] is not None]
        means.append(np.mean(layer_values))
        stds.append(np.std(layer_values, ddof=1))
    
    processed_data.append((means, stds))

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
x_ticks = np.arange(1, max(len(means) for means, stds in processed_data))

# Line plot for error bars (Layer 1 mean and std)
means, stds = zip(*[([x[0]], [y[0]]) for x, y in processed_data])
means = np.concatenate(means)
stds = np.concatenate(stds)
ax.errorbar(x_ticks[0:len(means)], means, yerr=stds, fmt='-o', label='Average', color='blue')

# Determine the number of groups and number of bars in each group
n_groups = len(processed_data)
n_bars = len(processed_data[0][0]) - 1  # Exclude the first bar which is plotted as error bar

# Calculate bar width
bar_width = 1 / (n_bars + 2)  # Adding space for separation
index = np.arange(n_groups) +0.9

for i in range(n_bars):
    means = [group[0][i+1] if len(group[0]) > i+1 else 0 for group in processed_data]
    stds = [group[1][i+1] if len(group[1]) > i+1 else 0 for group in processed_data]
    ax.bar(index + i * bar_width, means, bar_width, yerr=stds, label=f'Layer {i+2}')
        
ax.set_xlabel('Layer Index')
ax.set_ylabel('Accuracy')
ax.set_xticks(x_ticks)
ax.set_xticklabels([f'M{i}\n 1~{i}' for i in range(1, len(x_ticks) + 1)])
ax.legend()
plt.title('Layer Accuracy and Error Bars')
plt.tight_layout()
plt.show()
plt.savefig('./results/imagenet/predict.pdf')