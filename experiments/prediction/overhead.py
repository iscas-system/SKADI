#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# %%


import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.ticker as mtick

# mpl.use("Agg")

y_1 = [0.099, 0.106, 0.107, 0.113, 0.122]
# mean: 0.10939999999999998
print(np.array(y_1).mean())
x_1 = [i for i in range(len(y_1))]
colo_names = [1, 2, 4, 8, 16]
mpl.rcParams["hatch.linewidth"] = 1.0
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

ax.bar(
    x_1,
    y_1,
    0.3,
    # label="MLP-layer",
    color="none",
    hatch="\\\\\\\\\\\\\\\\",
    # edgecolor="#9F9F9F",
    edgecolor="#7EA6E0",
)

# plt.legend(ncol=4,  fontsize=16)
plt.xlabel("Batch Size", fontsize=18)
plt.ylabel("Overhead(ms)", fontsize=18)
plt.xticks(x_1, colo_names,  fontsize=14)
y_ticks = [0.0, 0.05, 0.1, 0.15]
# plt.set_yticklabels(y_ticks, rotation=0, fontsize=14)
plt.yticks(y_ticks,  fontsize=14)
plt.tight_layout()
plt.savefig("./prediction_overhead.png", bbox_inches="tight")
plt.show()


# %%
