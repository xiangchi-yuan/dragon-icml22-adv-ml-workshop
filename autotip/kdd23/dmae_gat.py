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
y_0 = [85.2, 59.08, 45.65, 20.9, 14.4, 12.56]
y_015 = [83.46, 60.32, 47.26, 23.88, 15.55, 12.69]
y_030 = [83.71, 61.44, 48.13, 26.37, 18.28, 14.43]
y_045 = [80.22, 61.82, 51.87, 31.59, 20.65, 16.92]
y_060 = [78.36, 65.05, 55.6, 41.79, 30.1, 22.76]

# 第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(ls='--')

plt.plot(x_label_rate, y_0, color='lightseagreen', marker='o', linestyle='--', label='w.o. DMAE')
plt.plot(x_label_rate, y_015, color='darkgoldenrod', marker='>', linestyle='dashed', label='DMAE=0.15')
plt.plot(x_label_rate, y_030, color='red', marker='>', linestyle='dashed', label='DMAE=0.3')
plt.plot(x_label_rate, y_045, color='orange', marker='>', linestyle='dashed', label='DMAE0.45')
plt.plot(x_label_rate, y_060, color='green', marker='>', linestyle='dashed', label='DMAE0.6')
# 添加文本 #x轴文本
plt.xlabel('Attack Level', fontsize=15)
# y轴文本
plt.ylabel('Accuracy(%)', fontsize=15)
# 标题

plt.xticks(size=13)
plt.yticks(size=12)
# 第3步：显示图形
plt.legend(loc='lower left', fontsize=10)
plt.savefig("C:/Users/yuanx/Desktop/fig/dmae-gat.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()
