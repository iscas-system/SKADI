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

# fname = "../../data/modeling/2in7results.csv"
fname = "/home/onceas/wanna/mt-dnn/result/2080Ti/validate/result.csv"
plain_data = np.array(pd.read_csv(fname, header=0).values.tolist())
# print(plain_data)
data = np.array([item for item in plain_data if "resnet34" not in item[0]])
print(data)
print(data.shape)
lens = data.shape[0]//2
mlp_operator_error = data[0:lens, 2].astype(np.float32)
print(np.mean(mlp_operator_error[0:lens]))
mlp_layer_error = data[lens:, 2].astype(np.float32)
print(np.mean(mlp_layer_error[0:lens]))
# lr_error = data[23:45, 3].astype(np.float32)
# print(np.mean(lr_error[0:21]))
# svm_error = data[45:, 3].astype(np.float32)
# print(np.mean(svm_error[0:28]))
# all_mlp_error = data[22, 3].astype(np.float32)
# print(mlp_error)

# colo_names = [
#     "(Res50,Res101)",
#     "(Res50,Res152)",
#     "(Res50,IncepV3)",
#     "(Res50,VGG16)",
#     "(Res50,VGG19)",
#     # "(Res50,bert)",
#     "(Res101,Res152)",
#     "(Res101,IncepV3)",
#     "(Res101,VGG16)",
#     "(Res101,VGG19)",
#     # "(Res101,Bert)",
#     "(Res152,IncepV3)",
#     "(Res152,VGG16)",
#     "(Res152,VGG19)",
#     # "(Res152,bert)",
#     "(IncepV3,VGG16)",
#     "(IncepV3,VGG19)",
#     # "(IncepV3,bert)",
#     "(VGG16,VGG19)",
#     # "(VGG16,bert)",
#     # "(VGG19,bert)",
#     "all",
# ]
plain_names = data[0:lens, 0]
colo_names = []
for col in plain_names:
    if "all" in col:
        colo_names.append(col)
        continue
    if "inception_v3" in col:
        col = col.replace("inception_v3", "incepV3")
    items = col.split("_")
    col = items[0]+"+"+items[1]
    colo_names.append(col)
print(colo_names)


x_0 = [i - 0.3 for i in range(lens)]
x_1 = [i for i in range(lens)]
x_2 = [i + 0.3 for i in range(lens)]
x_3 = [21.6]

mpl.rcParams["hatch.linewidth"] = 1.0
fig = plt.figure(figsize=(16, 3))
ax = fig.add_subplot(111)

print(mlp_layer_error)
print(mlp_operator_error)

ax.bar(
    x_1,
    mlp_layer_error,
    0.3,
    label="MLP-layer",
    color="none",
    hatch="\\\\\\\\\\\\\\\\",
    # edgecolor="#9F9F9F",
    edgecolor="#7EA6E0",
)
ax.bar(
    x_2,
    mlp_operator_error,
    0.3,
    label="MLP-operator",
    color="none",
    hatch="////////",
    edgecolor="#FF3333"
)
# ax.bar(
#     x_3,
#     all_mlp_error,
#     0.3,
#     color="none",
#     hatch="\\\\\\\\\\\\\\\\",
#     label="Cross Validation",
#     edgecolor="#138D8C",
# )
ax.set_xlim(-0.5, 15.75)
# plt.style.use('default')
plt.legend(ncol=4,  fontsize=16)
plt.ylabel("Prediction MAPE", fontsize=18)
plt.xticks(x_1, colo_names, rotation=30, fontsize=14)
y_ticks = [0.0, 0.1, 0.2]
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
# plt.set_yticklabels(y_ticks, rotation=0, fontsize=14)
plt.yticks(y_ticks, rotation=30, fontsize=14)
plt.tight_layout()
plt.savefig("./prediction_error.png", bbox_inches="tight")
plt.show()


# %%
