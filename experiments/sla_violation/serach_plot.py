
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.use("TkAgg")

# tcp
tcp_x = np.arange(1, 5.1, 0.1)
tcp_y = 2**(tcp_x-1)
tcp_x1 = [1, 2, 3, 4, 5]
tcp_y1 = [1, 2, 4, 8, 16]
tcp_x2 = [5, 6, 7, 8]
tcp_y2 = [16, 15, 14, 13]
plt.scatter(tcp_x1, tcp_y1, color='blue', s=15)
plt.plot(tcp_x, tcp_y, label='tcp search', color='blue')
plt.scatter(tcp_x2, tcp_y2, color='blue', s=15)
plt.plot(tcp_x2, tcp_y2, color='blue')

# linear
line_x = [i for i in range(1, 14)]
line_y = line_x
plt.scatter(line_x, line_y, color='orange', s=15)
plt.plot(line_x, line_y, label='linear search', color='orange')

# binary
binary_x = [1, 2]
binary_y = [9, 13]
plt.scatter(binary_x, binary_y, color='green', s=15)
plt.plot(binary_x, binary_y, label='binary search', color='green')

# threshold
target_x = np.arange(0, 18, 0.1)
target = [13 for i in range(len(target_x))]
plt.plot(target_x, target, label='threshold', color='red')

# 添加标签和标题
plt.xlabel('search time')
plt.ylabel('search position')
plt.title('search path')

plt.ylim(0, 17)
plt.xlim(0, 17)
# 显示图例
plt.legend()
# 显示图形
plt.show()

# %%
