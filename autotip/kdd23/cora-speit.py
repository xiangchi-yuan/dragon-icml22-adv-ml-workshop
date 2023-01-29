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
31.84,

]
y_gat = [84.95,
64.68,
52.99,
42.41,
35.2,
31.22,

]
y_rgcn = [85.57,
57.84,
42.29,
37.31,
32.84,
30.97,

]
y_gcnsvd = [64.76,
42.41,
36.44,
31.72,
31.22,
29.6,

]
y_gatguard = [81.22,
80.85,
81.09,
80.97,
81.34,
81.47,

]
y_gcn_at = [83.58,
82.34,
82.21,
79.48,
72.89,
62.31,

]
y_gat_at = [78.61,
78.86,
79.48,
78.86,
79.35,
78.48,

]

y_rgcn_at = [85.07,
82.96,
81.34,
79.85,
72.89,
50.12,

]
y_DRAGON = [85.03,
84.83,
84.49,
84.54,
84.7,
84.54,

]

y_DRAGON_at = [81.,
81.14,
81.09,
81.18,
80.22,
80.68,

]
# 第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(ls='--')


plt.plot(x_label_rate, y_gat, color='darkgoldenrod', marker='2', linestyle='dashed', label='GAT')
plt.plot(x_label_rate, y_gcn_at, color='grey', marker='^', linestyle='dashed', label='RGCN')
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
plt.savefig("C:/Users/yuanx/Desktop/fig/cora-speit.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()