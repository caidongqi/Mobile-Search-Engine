import matplotlib.pyplot as plt
# 创建一个新的图
# 假设 layer_num 是层的总数
layer_num=32
layer_keys = list(range(1, layer_num + 1))

# 打开文件
with open('model_accuracies.txt', 'r') as file:
    # 读取文件内容
    accuracies = file.read()

with open('layer_means.txt', 'r') as file:
    # 读取文件内容
    layer_means = file.read()

fig, ax1 = plt.subplots()

# 绘制准确率曲线
ax1.set_xlabel('Layer Key')
ax1.set_ylabel('Accuracy', color='tab:red')
ax1.plot(layer_keys, accuracies, 'r-', label='Accuracy')  # 'r-' 表示红色实线
ax1.tick_params(axis='y', labelcolor='tab:red')

# # 创建一个共享x轴的第二个轴用于绘制层均值
# ax2 = ax1.twinx()  # 共享x轴
# ax2.set_ylabel('Mean Layer', color='tab:blue')
# ax2.bar(layer_keys, layer_means, color='b', label='Mean Layer')  # 'b-' 表示蓝色实线
# ax2.tick_params(axis='y', labelcolor='tab:blue')


ax1.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
# ax2.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

# 显示图表
plt.title('Accuracy and Mean Layer per Layer Key')
plt.show()
plt.savefig('./results/flickr8k_lora_val_nohead/model_accuracy.pdf')
