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
y_gcn = [95.62,
95.22,
92.01,
81.16,
58.72,
34.31,

]

y_gat = [95.92,
95.6,
94.62,
93.44,
92.56,
91.26,

]
y_rgcn = [95.54,
93.54,
84.48,
71.88,
55.15,
39.59,

]


y_gcn_at = [95.83,
94.69,
83.3,
57.23,
33.63,
17.52,

]

y_gat_at = [95.48,
95.36,
95.01,
94.58,
94.29,
93.77,
]

y_rgcn_at = [95.77,
92.76,
80.77,
59.5,
36.01,
20.05,

]
y_DRAGON = [96.23,
96,
95.71,
95.37,
95.24,
94.91,

]

y_DRAGON_at = [95.61,
95.52,
95.5,
95.46,
95.42,
95.35,

]
# 第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(ls='--')


plt.plot(x_label_rate, y_rgcn, color='darkgoldenrod', marker='2', linestyle='--', label='RGCN')
plt.plot(x_label_rate, y_gat, color='purple', marker='3', linestyle='--', label='GAT')
plt.plot(x_label_rate, y_gcn_at, color='grey', marker='^', linestyle='dashed', label='GCN-AT')
plt.plot(x_label_rate, y_gat_at, color='blue', marker='o', linestyle='dashed', label='GAT-AT')
plt.plot(x_label_rate, y_rgcn_at, color='green', marker='>', linestyle='dashed', label='RGCN-AT')
plt.plot(x_label_rate, y_DRAGON, color='red', marker='*', linestyle='--', label='Ours')
plt.plot(x_label_rate, y_DRAGON_at, color='orange', marker='*', linestyle='--', label='Ours-AT')

# 添加文本 #x轴文本
plt.xlabel('Attack Level', fontsize=15)
# y轴文本
plt.ylabel('Accuracy(%)', fontsize=15)
# 标题

plt.xticks(size=13)
plt.yticks(size=12)
# 第3步：显示图形
plt.legend(loc='lower left', fontsize=12)
plt.savefig("C:/Users/yuanx/Desktop/fig/reddit-pgd.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()