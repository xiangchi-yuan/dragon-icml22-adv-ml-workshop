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
95.25,
93.87,
90.65,
86.11,
80.06,

]

y_gat = [95.92,
95.63,
95.18,
94.44,
93.56,
92.58,

]
y_rgcn = [95.54,
93.49,
84.52,
72.23,
56.62,
40.98,

]


y_gcn_at = [95.83,
94.84,
88.61,
78.02,
64.04,
50.17,

]

y_gat_at = [95.48,
95.4,
95.27,
95.15,
95.13,
95.08,

]

y_rgcn_at = [95.77,
92.75,
81.06,
60.59,
38.36,
21.89,


]
y_DRAGON = [96.23,
96.07,
95.95,
95.85,
95.79,
95.69,

]

y_DRAGON_at = [95.6,
95.53,
95.54,
95.53,
95.51,
95.51,
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
plt.plot(x_label_rate, y_gcn_at, color='green', marker='>', linestyle='dashed', label='GCN-AT')
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
plt.savefig("C:/Users/yuanx/Desktop/fig/reddit-speit.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()