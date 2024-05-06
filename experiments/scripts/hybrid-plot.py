
# %%

import csv
import torch.nn as nn
import abacus.modeling.predictor.mlp as mlp
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import os
import glob
# from abacus.option import RunConfig
import json
from abacus.modeling.utils import AverageMeter
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

model_names = {
    "resnet50": 18,
    "resnet101": 35,
    "resnet152": 52,
    "inception_v3": 14,
    "vgg16": 19,
    "vgg19": 22,
    "resnet18": 10,
    "resnet34": 18,
    "googlenet": 17,
    "vgg11": 14,
    "vgg13": 16,
}


def load_single_file(filepath):
    print("load file:", filepath)
    data = pd.read_csv(filepath, header=0)
    data = data.values.tolist()
    total_data_num = len(data)
    print("{} samples loaded from {}".format(total_data_num, filepath))
    data = np.array(data)
    single1 = {}
    for line in data:
        single1[int(line[3])] = float(line[-2])
    return single1


def load_comb_file(filepath):
    print("load file:", filepath)
    data = pd.read_csv(filepath, header=0)
    data = data.values.tolist()
    total_data_num = len(data)
    print("{} samples loaded from {}".format(total_data_num, filepath))
    data = np.array(data)
    co_located = []
    bs_dict = {}
    for line in data:
        bs1 = int(line[3])
        bs2 = int(line[8])
        co_located.append((bs1, bs2, float(line[-2])))
        bs_dict[100*bs1+bs2] = float(line[-2])
    return co_located, bs_dict


if __name__ == '__main__':
    dir_path = "/home/onceas/wanna/mt-dnn/data/profile/2080Ti/2in7"
    combs = ["resnet101_resnet152"]
    combs = ["resnet152_vgg16"]
    # combs = ["vgg16_vgg19"]
    idx = 1
    fig = plt.figure(figsize=(20, 10))
    for comb in combs:
        # filepath1 = "{}/resnet101.csv".format(dir_path, comb)
        filepath1 = "{}/resnet152.csv".format(dir_path, comb)
        # filepath1 = "{}/vgg16.csv".format(dir_path, comb)
        filepath2 = "{}/vgg16.csv".format(dir_path, comb)
        filepath3 = "{}/{}.csv".format(dir_path, comb)
        idx += 1
        single1 = load_single_file(filepath1)
        single2 = load_single_file(filepath2)
        co_located, bs_dict = load_comb_file(filepath3)
        print(single1)
        print(single2)
        print(co_located)
        print(bs_dict)
        axl = fig.add_subplot(111, projection='3d')

        max_bs = 32
        x = [i for i in range(1, max_bs+1)]
        y = [i for i in range(1,  max_bs+1)]
        x, y = np.meshgrid(x, y)
        z = []
        for i in range(1,  max_bs+1):
            tmp = []
            for j in range(1,  max_bs+1):
                tmp.append(bs_dict[100*i+j]/(single1[i] + single2[j]))
            z.append(tmp)
        z = np.array(z)
        print(x)
        print(x.shape)
        print(y.shape)
        print(z.shape)
        print(z[z > 1.0])
        # z1 = np.ones((max_bs, max_bs))
        z1 = np.full((max_bs, max_bs), 0.9)
        axl.plot_surface(x, y, z, rstride=1,
                         cstride=1, cmap='rainbow')
        axl.plot_surface(x, y, z1, rstride=1,
                         cstride=1, color='gray', alpha=0.3)
        axl.set_xlim(1, max_bs)
        axl.set_ylim(1, max_bs)

        # 找到两个曲面的交点
        intersection = []
        for i in range(1,  max_bs+1):
            tmp = []
            for j in range(1,  max_bs+1):
                num = bs_dict[100*i+j]/(single1[i] + single2[j])-0.9
                if (num > 0 and num <= 0.01):
                    intersection.append(
                        [i, j, num+0.904])
        # intersection = np.array([[32, 32, 1]])
        intersection = np.array(intersection)
        print(intersection)
        # 绘制交点
        axl.scatter(intersection[:, 0], intersection[:, 1],
                    intersection[:, 2], color='purple', s=10)

        axl.invert_xaxis()
        axl.set_xlabel('model1 bs', fontsize=15)
        axl.set_ylabel('model2 bs', fontsize=15)
        axl.set_zlabel('ratio', fontsize=15)
    fig.tight_layout()
    plt.show()

# %%
