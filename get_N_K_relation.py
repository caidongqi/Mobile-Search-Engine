import csv
import matplotlib.pyplot as plt

# 读取CSV文件
filename = "end_to_end_lora_N_K.csv"
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
r_values = [1,5,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 300, 400, 500, 600, 700, 800, 900, 1000]
values=[1,10,30,50,100,130,300]
# 绘制折线图
plt.figure(figsize=(10, 6))


for i, r in enumerate(r_values):
    if r not in values:
        continue 
    total_accuracies = [[total_data[k][i] if i < len(total_data[k]) else None] for k in sorted(total_data.keys())]
    dynamic_accuracies = [[dynamic_data[k][i] if i < len(dynamic_data[k]) else None] for k in sorted(dynamic_data.keys())]

    # 绘制 total_accuracies 折线，设置为红色
    plt.plot(list(sorted(total_data.keys())), total_accuracies, label=f'total r@{r}')

    # 绘制 dynamic_accuracies 折线，设置为蓝色
    #plt.plot(list(sorted(dynamic_data.keys())), dynamic_accuracies, color='blue', label=f'dynamic r@{r}')

    # # 添加文本标注
    # for k in sorted(total_data.keys()):
    #     if total_accuracies[k] is not None:
    #         plt.text(k, total_accuracies[k], f'r@{r}', fontsize=8, color='red', ha='center', va='bottom')
    # for k in sorted(dynamic_data.keys()):
    #     if dynamic_accuracies[k] is not None:
    #         plt.text(k, dynamic_accuracies[k], f'r@{r}', fontsize=8, color='blue', ha='center', va='bottom')

# for i, r in enumerate(r_values):
#     total_accuracies = [[total_data[k][i] if i < len(total_data[k]) else None ] for k in sorted(total_data.keys())]

#     #total_accuracies = [if (total_data[k][i]): total_data[k][i] else [] for k in sorted(total_data.keys())]
#     dynamic_accuracies=[[dynamic_data[k][i] if i< len(dynamic_data[k]) else None ] for k in sorted(dynamic_data.keys())]
#     #dynamic_accuracies = [dynamic_data[k][i] for k in sorted(dynamic_data.keys())]
#     plt.plot(list(sorted(total_data.keys())), total_accuracies, label=f'total r@{r}')
#     plt.plot(list(sorted(dynamic_data.keys())), dynamic_accuracies, label=f'dynamic r@{r}')

#     # plt.plot(r, total_accuracies)
#     # plt.plot(r, dynamic_accuracies)

plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K for different r@k')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('accuracy_vs_k.png')
