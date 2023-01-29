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
y_gcn = [85.2,
63.93,
55.72,
46.27,
35.95,
31.84
]
y_gat = [84.95,
64.68,
52.99,
42.41,
35.2,
31.22,
]
y_rgcn = [85.57,
55.72,
39.55,
34.83,
33.33,
33.83,

]
y_gcnsvd = [64.76,
42.41,
36.44,
31.72,
31.22,
29.6,
]
y_gatguard = [81.22,
81.09,
81.22,
80.85,
80.97,
80.47,

]
y_gcn_at = [83.58,
77.24,
70.4,
67.29,
66.66,
59.2,

]
y_gat_at = [78.61,
78.61,
79.48,
78.36,
78.86,
78.35,

]

y_rgcn_at = [85.07,
82.71,
79.98,
78.48,
78.73,
78.48,

]
y_DRAGON = [84.91,
85.32,
84.62,
85.12,
84.08,
84.04,

]

y_DRAGON_at = [81.43,
81.01,
80.1,
81.59,
81.01,
80.35,

]
# 第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(ls='--')


plt.plot(x_label_rate, y_gat, color='darkgoldenrod', marker='2', linestyle='dashed', label='GAT')

plt.plot(x_label_rate, y_gatguard, color='purple', marker='3', linestyle='dashed', label='GATGuard')
plt.plot(x_label_rate, y_gcn_at, color='grey', marker='^', linestyle='dashed', label='GCN-AT')
plt.plot(x_label_rate, y_gat_at, color='blue', marker='o', linestyle='dashed', label='GAT-AT')
plt.plot(x_label_rate, y_rgcn_at, color='green', marker='>', linestyle='dashed', label='RGCN-AT')
plt.plot(x_label_rate, y_DRAGON, color='red', marker='*', linestyle='dashed', label='Ours')
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
plt.savefig("C:/Users/yuanx/Desktop/fig/cora-pgd.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()