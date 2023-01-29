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
y_gcn = [84.25,
65.67,
57.38,
33.62,
17.91,
12.48,

]

error_gcn = [0.6, 3.23, 3.01, 3.04, 0.81, 0.16]

y_gcnsvd = [
64.80,
44.15,
42.00,
37.35,
33.71,
33.00,
]
error_gcnsvd = [1.03, 0.47, 2.02, 2.99, 1.43, 0.41, 1.22]
y_gat = [83.62,
58.96,
38.93,
24.3,
15.84,
14.3,

]
error_gat = [0.51, 4.32, 3.78, 4.38, 2.52, 1.68]
y_rgcn = [84.04,
51.84,
23.29,
13.68,
12.06,
11.98,

]
error_rgcn = [0.71, 1.5, 0.71, 0.37, 0.1, 0.06]

y_gcnguard = [78.48,
78.86,
78.48,
78.48,
78.48,
77.99,
]


y_gat_at = [78.61,
78.86,
79.48,
78.86,
79.35,
78.48,

]
error_gat_at = [0.98, 0.85, 0.79, 0.85, 0.16, 0.68]
y_rgcn_at = [85.07,
79.96,
72.34,
49.85,
25.89,
24.12,

]
y_DRAGON = [85.03,
84.83,
84.49,
84.54,
84.7,
84.54,

]
error_dragon =[0.75, 0.6, 0.31, 0.52, 0.36, 0.31]

y_gcn_at = [83.54,
81.71,
79.65,
71.47,
47.76,
35.60
]
error_gcn_at = [0.16, 1.18, 0.88, 4.74, 3.91, 4.47]
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