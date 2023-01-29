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
y_0 = [79.68,79.52,79.48,79.27,78.48,78.36]
y_01 = [81.56,
81.24,
81.01,
80.51,
79.68,
78.85]
y_03 = [81.89,
81.34,
80.39,
80.21,
80.02,
79.68]

y_09 = [81.22,
80.64,
80.48,
80.35,
79.93,
79.89]
y_moe10_1 = [83.08,
82.75,
81.67,
81.67,
81.22,
80.76,
]
# 第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(ls='--')

plt.plot(x_label_rate, y_0, color='lightseagreen', marker='o', linestyle='--', label='GAT')
plt.plot(x_label_rate, y_01, color='darkgoldenrod', marker='>', linestyle='dashed', label='DP rate=0.1')
plt.plot(x_label_rate, y_03, color='grey', marker='>', linestyle='dashed', label='DP rate=0.3')
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
plt.legend(loc='lower left', fontsize=10)
plt.savefig("C:/Users/yuanx/Desktop/fig/dprate_AT_cora.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()
