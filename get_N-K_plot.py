import csv
import matplotlib.pyplot as plt

# 读取CSV文件
filename = "pc/Mobile-Search-Engine/end_to_end_lora_N_K.csv"
data = []
with open(filename, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

# 初始化数据字典
total_data = {}
dynamic_data = {}

# 处理数据
for row in data:
    if row[0] == 'total':
        k = int(row[2])
        accuracies = [float(x) for x in row[3:]]
        total_data[k] = accuracies
    elif row[0] == 'dynamic':
        k = int(row[2])
        accuracies = [float(x) for x in row[3:]]
        dynamic_data[k] = accuracies

# 提取不同r@k的准确率数据
r_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 300, 400, 500, 600, 700, 800, 900, 1000]

# 绘制折线图
plt.figure(figsize=(10, 6))

for r in r_values:
    total_accuracies = [total_data[k][r - 1] for k in sorted(total_data.keys())]
    dynamic_accuracies = [dynamic_data[k][r - 1] for k in sorted(dynamic_data.keys())]
    plt.plot(list(sorted(total_data.keys())), total_accuracies, label=f'total r@{r}')
    plt.plot(list(sorted(dynamic_data.keys())), dynamic_accuracies, label=f'dynamic r@{r}')

plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K for different r@k')
plt.legend()
plt.grid(True)
plt.show()
