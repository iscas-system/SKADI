#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import glob
import pandas as pd
import os


mpl.rcParams["font.family"] = "Times New Roman"


def load_all_unpred_latency(
    data_path="../data/profile/2080Ti/2in7-unpre-batch",
    # data_path="../data/profile/2080Ti/2in7",
):
    all_unpred_latency = dict()
    for filename in glob.glob(os.path.join(data_path, "*.csv")):
        # filepath = os.path.join(data_path, filename)
        data = pd.read_csv(filename, header=0)
        data = data.values.tolist()
        total_data_num = len(data)
        print("{} samples loaded from {}".format(total_data_num, filename))
        data = np.array(data)
        n = data.shape[0]
        unpred_latency = []
        for i in range(n):
            line = data[i]
            # print(line)
            unpred_latency.append(line[-2].astype(float) / 1000)
        unpred_latency = np.array(unpred_latency)
        # if all_unpred_latency is None:
        #     all_unpred_latency = unpred_latency
        # else:
        #     # print(all_unpred_latency.shape)
        #     # print(unpred_latency.shape)
        #     all_unpred_latency = np.concatenate(
        #         (all_unpred_latency, unpred_latency), axis=0
        #     )
        all_unpred_latency[filename.split(
            '/')[-1]] = unpred_latency.astype(float)
    return all_unpred_latency


all_unpred_latency = load_all_unpred_latency()
print(all_unpred_latency.keys())


def find_key(model_name):
    for k in all_unpred_latency.keys():
        if model_name.lower() in k:
            return k


fig = plt.figure(figsize=(10, 3))
ax = fig.add_subplot(111)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set_xlabel("Latency(ms)", fontsize=20)
ax.set_ylabel("CDF", fontsize=20)
ax.tick_params("x", labelsize=18)
ax.tick_params("y", labelsize=18)

# df = pd.DataFrame(list(all_unpred_latency.items()), columns=['type', 'data'])
# print(df)
# sns.ecdfplot(data=df, x='data', hue='type')

sns.ecdfplot(data=all_unpred_latency[find_key(
    "Resnet18")], label="Resnet18")
sns.ecdfplot(data=all_unpred_latency[find_key(
    "Resnet34")], label="Resnet34")
sns.ecdfplot(data=all_unpred_latency[find_key(
    "Resnet50")], label="Resnet50")
sns.ecdfplot(all_unpred_latency[find_key(
    "Resnet101")], label="Resnet101")
# sns.ecdfplot(all_unpred_latency[3], label="Resnet152")
sns.ecdfplot(all_unpred_latency[find_key(
    "Inception_v3")], label="Inception_v3")
sns.ecdfplot(all_unpred_latency[find_key("VGG16")], label="VGG16")
sns.ecdfplot(all_unpred_latency[find_key("VGG19")], label="VGG19")
# sns.ecdfplot(all_unpred_latency[4], label="Bert")
plt.legend(fontsize=16, ncol=2, loc=(0.5, 0), frameon=False)
plt.savefig("../figure/unpredictable_latency.pdf", bbox_inches="tight")
plt.savefig("../figure/unpredictable_latency.png", bbox_inches="tight")
plt.show()

for k in all_unpred_latency.keys():
    data = all_unpred_latency[k]
    print("{} min:{} max:{} ".format(k, np.min(data), np.max(data)))

# print(np.sum(all_unpred_latency > 30))
