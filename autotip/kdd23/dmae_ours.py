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
y_0 = [84.7, 71.39, 59.2, 54.1, 44.15, 34.2]
y_015 = [84.08, 72.51, 60.57, 58.08, 48.13, 38.81]
y_030 = [83.14,72.01,63.81,61.69,56.22,49.25]
y_045 = [82.09,70.65,63.43,62.06,57.21,49.13]
y_060 = [80.6,70.7,65.42,62.56,59.83,55.1]

# 第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(ls='--')

plt.plot(x_label_rate, y_0, color='lightseagreen', marker='o', linestyle='--', label='w.o. DMAE')
plt.plot(x_label_rate, y_015, color='darkgoldenrod', marker='>', linestyle='dashed', label='DMAE=0.15')
plt.plot(x_label_rate, y_030, color='red', marker='>', linestyle='dashed', label='DMAE=0.3')
plt.plot(x_label_rate, y_045, color='orange', marker='>', linestyle='dashed', label='DMAE=0.45')
plt.plot(x_label_rate, y_060, color='green', marker='>', linestyle='dashed', label='DMAE=0.6')
# 添加文本 #x轴文本
plt.xlabel('Attack Level', fontsize=15)
# y轴文本
plt.ylabel('Accuracy(%)', fontsize=15)
# 标题

plt.xticks(size=13)
plt.yticks(size=12)
# 第3步：显示图形
plt.legend(loc='lower left', fontsize=12)
plt.savefig("C:/Users/yuanx/Desktop/fig/dmae-ours.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()
