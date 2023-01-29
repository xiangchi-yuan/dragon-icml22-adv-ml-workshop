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
60.19,
40.96,
27.06,
20.38,
19.33,


]
y_gat = [72.94,
21.53,
15.46,
15.67,
15.88,
15.46,

]
y_rgcn = [71.79,
47.41,
44.51,
39.08,
37.51,
37.51,

]
y_gcnsvd = [37.48,
24.35,
22.36,
20.9,
20.38,
19.64,
]
y_gatguard = [72.41,
72.41,
72.41,
72.41,
72.41,
71.06,
]
y_gcn_at = [72.94,
40.75,
31.66,
33.33,
25.71,
28,

]
y_gat_at = [72.94,
63.32,
50.05,
37.62,
29.68,
31.87,

]

y_rgcn_at = [74.92,
71.58,
67.61,
63.85,
61.44,
57.68,

]
y_DRAGON = [75.55,
74.96,
75.27,
75.58,
75.55,
74.4,

]

y_DRAGON_at = [70.92,
70.92,
70.18,
70.81,
70.78,
70.08,


]
# 第2步：使用plot绘制线条 第1个参数是x的坐标值，第2个参数是y的坐标值

#
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(ls='--')


plt.plot(x_label_rate, y_rgcn, color='darkgoldenrod', marker='2', linestyle='--', label='RGCN')
plt.plot(x_label_rate, y_gcn_at, color='grey', marker='^', linestyle='dashed', label='GCN-AT')
plt.plot(x_label_rate, y_gatguard, color='purple', marker='3', linestyle='--', label='GATGuard')
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
plt.savefig("C:/Users/yuanx/Desktop/fig/citeseer-speit.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()