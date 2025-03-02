import matplotlib.pyplot as plt
import numpy as np
# 假设 layers 是你的数组
S=[1,5, 10, 20, 30, 40, 50, 60,70,80,90,100,110,120,130,150,160,170,180,200,210,220,230,240,250,260,270,280,300,400,500,600]
for i in S:
    ground_truth=f'results/clotho_head/R{i}/layers.txt'

    layers=np.loadtxt(ground_truth)
    print(i)
    print(np.mean(layers))