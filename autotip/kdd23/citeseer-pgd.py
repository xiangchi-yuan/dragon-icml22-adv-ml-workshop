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
y_gcn = [70.95,
27.8,
47.65,
27.06,
26.44,
27.48,

]
y_gat = [72.94,
19.02,
16.93,
17.66,
15.78,
14.21,
]
y_rgcn = [71.79,
58.62,
38.85,
34.8,
33.86,
32.18,
]
y_gcnsvd = [37.48,
25.36,
21.84,
20.9,
20.48,
19.64,

]
y_gatguard = [72.41,
72.41,
72.41,
71.37,
63.43,
57.16,

]
y_gcn_at = [72.94,
40.75,
31.66,
33.33,
25.71,
28,

]
y_gat_at = [71.37,
70.43,
70.32,
69.7,
69.49,
68.86,

]

y_rgcn_at = [74.92,
72,
65.52,
63.22,
58.93,
52.66,

]
y_DRAGON = [75.3,
75.2,
74.99,
73.84,
70.5,
66.11,

]

y_DRAGON_at = [71.13,
70.43,
70.22,
69.77,
70.57,
70.22,

]
# 第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(ls='--')


plt.plot(x_label_rate, y_rgcn, color='darkgoldenrod', marker='2', linestyle='dashed', label='RGCN')
plt.plot(x_label_rate, y_gcn_at, color='grey', marker='^', linestyle='dashed', label='GCN-AT')
plt.plot(x_label_rate, y_gatguard, color='purple', marker='3', linestyle='dashed', label='GATGuard')
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
plt.savefig("C:/Users/yuanx/Desktop/fig/citeseer-pgd.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()