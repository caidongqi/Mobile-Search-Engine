import pandas as pd
import matplotlib.pyplot as plt

# 读取xlsx文件
df = pd.read_excel('topk-clotho.xlsx',sheet_name='Sheet0')

# 按照audio层数进行排序
df_sorted = df.sort_values(by='audio层数')

# 提取需要绘制的数据
audio_layers = df_sorted['audio层数']
counts_values = df_sorted[['counts_r1', 'counts_r5', 'counts_r10', 'counts_r20', 'counts_r30', 'counts_r40', 'counts_r50', 'counts_r60', 'counts_r70', 'counts_r80', 'counts_r90', 'counts_r100', 'counts_r110', 'counts_r120', 'counts_r130', 'counts_r300', 'counts_r400', 'counts_r500', 'counts_r600', 'counts_r700', 'counts_r800', 'counts_r900', 'counts_r1000']]

# 绘制曲线图
plt.figure(figsize=(10, 6))
handles = []
# 循环绘制每个counts值的曲线
for col in counts_values.columns:
    #plt.plot(audio_layers, counts_values[col], label=col)
    line, = plt.plot(audio_layers, counts_values[col], label=col)
    handles.append(line)
# 添加图例和标签
plt.xlabel('audio layer')
plt.ylabel('r@n')
plt.title('audio layer vs r@n')
# 调整图形布局，将图形放在靠左的位置
plt.subplots_adjust(left=0.1, right=0.7)

# 创建图例并放置在右侧
plt.legend(handles[::-1], counts_values.columns[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
#plt.legend(loc='center left', bbox_to_anchor=(0.5, 0.5))
plt.grid(True)
# 保存图形
plt.savefig('counts_vs_audio_layers.png')
# 显示图形
plt.show()
