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

y_gcnsvd = [
64.80,
40.67,
37.19,
34.70,
32.71,
29.98,
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
y_gcnguard = [78.48,
78.86,
78.48,
78.48,
78.48,
77.99,
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

y_gcn_at = [83.58,
83.21,
79.85,
70.27,
47.76,
29.60
]
# 第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(ls='--')

plt.plot(x_label_rate, y_gcn, color='purple', marker='o', linestyle='dashed', label='GCN')
plt.plot(x_label_rate, y_gat, color='darkgoldenrod', marker='2', linestyle='dashed', label='GAT')
plt.plot(x_label_rate, y_rgcn, color='grey', marker='^', linestyle='dashed', label='RGCN')
# plt.plot(x_label_rate, y_gcnguard, color='purple', marker='3', linestyle='dashed', label='GCNGuard')
plt.plot(x_label_rate, y_gcnsvd, color='lightblue', marker='2', linestyle='dashed', label='GCNSVD')
plt.plot(x_label_rate, y_gcn_at, color='orange', marker='1', linestyle='--', label='GCN-AT')

plt.plot(x_label_rate, y_rgcn_at, color='green', marker='>', linestyle='dashed', label='RGCN-AT')
plt.plot(x_label_rate, y_DRAGON, color='red', marker='*', linestyle='dashed', label='Ours')


# 添加文本 #x轴文本
plt.xlabel('Attack Level', fontsize=15)
# y轴文本
plt.ylabel('Accuracy(%)', fontsize=15)
# 标题

plt.xticks(size=13)
plt.yticks(size=12)
# 第3步：显示图形
plt.legend(loc='lower left', fontsize=12)
plt.savefig("C:/Users/yuanx/Desktop/fig/intro-cora.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()