import json
import matplotlib.pyplot as plt
import sys
import os
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt

# 3 给图片提阿甲注释和标题等

# 第1步：定义x和y坐标轴上的点  x坐标轴上点的数值
x_label_rate = [0, 1, 2, 3, 4, 5]
# y坐标轴上点的数值
y_0 = [84.48,64.68,53.18,24.28,18.58,16.64]
y_01 = [85.32,68.66,52.61,52.86,32.34,22.01]
y_03 = [85.32,74.38,63.56,43.53,32.21,25.5]
y_05 = [83.83,73.13,62.56,54.35,45.52,37.44]
y_07 = [84.08,70.77,59.83,60.32,48.26,36.82]
y_09 = [83.33,69.65,56.22,51.24,42.29,40.8]
y_moe10_1 = [86.07,68.3,54.88,50.37,44.28,36.82]
# 第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(ls='--')

plt.plot(x_label_rate, y_0, color='lightseagreen', marker='o', linestyle='--', label='w.o. DP')
plt.plot(x_label_rate, y_01, color='darkgoldenrod', marker='>', linestyle='dashed', label='DP rate=0.1')
plt.plot(x_label_rate, y_03, color='grey', marker='>', linestyle='dashed', label='DP rate=0.3')
plt.plot(x_label_rate, y_05, color='orange', marker='>', linestyle='dashed', label='DP rate=0.5')
plt.plot(x_label_rate, y_07, color='purple', marker='>', linestyle='dashed', label='DP rate=0.7')
plt.plot(x_label_rate, y_09, color='green', marker='>', linestyle='dashed', label='DP rate=0.9')
plt.plot(x_label_rate, y_moe10_1, color='red', marker='*', linestyle='dashed', label='MoEDP')
# 添加文本 #x轴文本
plt.xlabel('Attack Level', fontsize=15)
# y轴文本
plt.ylabel('Accuracy(%)', fontsize=15)
# 标题

plt.xticks(size=13)
plt.yticks(size=12)
# 第3步：显示图形
plt.legend(loc='lower left', fontsize=12)
plt.savefig("C:/Users/yuanx/Desktop/fig/dprate_cora.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()