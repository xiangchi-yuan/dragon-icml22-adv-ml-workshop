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
y_gcn = [50.93,
44.21,
15.42,
7.94,
7.48,
7.44,

]
y_gat = [52.86,
38.44,
26.2,
10.25,
9.05,
8.76,
]
y_rgcn = [49.39,
49.28,
44.94,
45.47,
46.18,
45.65,
]


y_gcn_at = [45.65,
42.66,
39.12,
14.3,
10.2,
9.59,
]

y_gat_at = [
46.14,
45.67,
45.03,
43.8,
43.88,
44.07,

]

y_rgcn_at = [48.78,
44.38,
42.34,
42.35,
42.41,
37.98,
]
y_DRAGON = [53.15,
52,
51.74,
51.93,
52.09,
52.42,

]

y_DRAGON_at = [51.01,
51.29,
50.2,
49.69,
48.82,
47.95,

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
plt.savefig("C:/Users/yuanx/Desktop/fig/flickr-speit.pdf", bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
plt.show()