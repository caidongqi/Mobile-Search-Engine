import matplotlib.pyplot as plt
import numpy as np

s = [10,30, 50,60,70,80, 100] # target label: r@s==1
#layers = [28, 25, 21, 16] 
layers_S=[28.04,26.52,24.18,22.6104,20.9232,21.2404,5.97]
latency = [643.57 / 32 * i for i in layers_S] # total of 1000 images TODO: 看看一个图片的latency是多少
K=[10,30,50,80,100,130,300]
r1_K=[0.7244,0.7428,0.7456,0.7468,0.7472,0.7476,0.7476]

r1_S=[0.7472,0.7472,0.7448,0.7492,0.7396,0.7344,  0.2792]
r1_S1=[0.7616,0.7616,0.7592,0.7492,0.7396,0.7344,0.282]

r1_S_dynamic=[0.5132,0.524,0.4508,0.3484,0.2344,0.2852,  0.0936]
r10_k=[]
r1 = [0.7472, 0.7296, 0.6912, 0.6712] # with p_lora ;fine-grained k = 100
r1_plora = [0.5044, 0.3468, 0.0824, 0.0164]
r1_lora = [0.7308,0.6592, 0.542, 0.2284]
r1_zero_shot = [0.234, 0.1388, 0.006, 0.0016]

full_zero_shot = 0.7241 # zzl: 我这里记录的是0.7184
full_lora = 0.7676


X_R=['R1','R5','R10','R20','R30','R40','R50','R60','R70','R80','R90','R100']
Y_R_total=[0.7472,0.9316,0.9676,0.9832,0.9868,0.9888,0.99,0.9904,0.9908,0.992,0.9924,0.9924]
Y_R_dynamic=[0.5816,0.85,0.908,0.95,0.9696,0.9756,0.9832,0.988,0.9896,0.9904,0.9916,0.9924]
Y_R_zero_shot=[0.7245,0.90844,0.93984,0.96064,0.97054,0.9762,0.97998,0.98312,0.98532,0.98722,0.98876,0.98972]

plt.figure()
# # # plt.scatter(latency, r1, label='Ours', color='red', marker='o')
plt.scatter(latency, r1_S, label='total', color='red', marker='o')
plt.scatter(latency, r1_S_dynamic, label='dynamic', color='blue', marker='x')
# # # plt.scatter(latency, r1_lora, label='lora', color='green', marker='^')
# # # plt.scatter(latency, r1_zero_shot, label='zero-shot', color='black', marker='s')
# # # plt.scatter(643.57, full_zero_shot, label=' full_zero_shot', color='purple', marker='>')
# # # plt.scatter(643.57, full_lora, label=' full_lora', color='pink', marker='<')
plt.xlabel('latency')
plt.ylabel('R@1')

plt.legend()


# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# plt.rcParams['font.family'] = 'SimHei'

# plt.figure()
# plt.plot(X_R, Y_R_total, label='端到端系统', color='red', marker='o')
# plt.plot(X_R, Y_R_dynamic, label='动态筛选', color='green', marker='o')
# plt.plot(X_R, Y_R_zero_shot, label='基准模型', color='blue', marker='o')
# plt.xlabel('R@N')
# plt.ylabel('Accuracy')
# # Adding legend with Chinese characters using a built-in font
# # font = FontProperties(fname=plt.matplotlib.get_data_path() + '/fonts/ttf/msyh.ttf')  # SimSun is a common font for Chinese characters
# # plt.legend(prop=font)
# plt.legend()
plt.savefig('./results/imagenet/S_R1.pdf')