#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# %%
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl


def get_throughout(data):
    n = data.shape[0]
    latency_data = []
    for i in range(n):
        line = data[i]
        latency = [line[0], [float(line[2]), float(line[5]),
                   float(line[8]), float(line[11]), float(line[14]), float(line[17]), float(line[20]), float(line[23])]]
        latency_data.append(latency)
    latency_data = np.array(latency_data)
    return latency_data


def get_latency_tail(data):
    n = data.shape[0]
    latency_data = []
    for i in range(n):
        line = data[i]
        latency = [line[0], [float(line[1]), float(line[4]),
                   float(line[7]), float(line[10]), float(line[13]), float(line[16]), float(line[19]), float(line[22])]]
        latency_data.append(latency)
    latency_data = np.array(latency_data)
    return latency_data


def load_single_file(filepath):
    data = pd.read_csv(filepath, header=0)
    data = data.values.tolist()
    total_data_num = len(data)
    # print("{} samples loaded from {}".format(total_data_num, filepath))
    data = np.array(data)
    return get_throughout(data), get_latency_tail(data)


def plot_throu_with_load(throu_dict: dict, fixed_load: int, figure_type: str):
    mpl.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(12, 8))
    idx = 1
    fixed_load_dict = {}
    for items in throu_dict.items():
        sorted_items = sorted(items[1].items(), key=lambda x: int(x[0]))
        for item in sorted_items:
            if int(item[0]) == fixed_load:
                fixed_load_dict[items[0]] = item[1]
                break
    print(fixed_load_dict)
    colo_names = fixed_load_dict.keys()
    print(colo_names)

    axs = fig.add_subplot(1, 1, 1)
    idx += 1
    x_0 = [i for i in range(1, len(colo_names)+1)]
    y_1 = np.array([fixed_load_dict[it][0] for it in colo_names])
    y_2 = np.array([fixed_load_dict[it][1] for it in colo_names])
    y_3 = np.array([fixed_load_dict[it][2] for it in colo_names])
    y_4 = np.array([fixed_load_dict[it][3] for it in colo_names])
    y_5 = np.array([fixed_load_dict[it][4] for it in colo_names])
    y_6 = np.array([fixed_load_dict[it][5] for it in colo_names])
    y_8 = np.array([fixed_load_dict[it][7] for it in colo_names])
    print(y_8)
    y_1 /= 1000/fixed_load
    y_2 /= 1000/fixed_load
    y_3 /= 1000/fixed_load
    y_4 /= 1000/fixed_load
    y_5 /= 1000/fixed_load
    y_6 /= 1000/fixed_load
    y_8 /= 1000/fixed_load

    if figure_type == "plot":
        axs.plot(x_0, y_1, label="FCFS", color="#6F0000")
        axs.plot(x_0, y_2, label="SJF", color="#333333")
        axs.plot(x_0, y_3, label="EDF", color="#999999")
        axs.plot(x_0, y_4, color="blue", label="Abacus")
        # axs.plot(x_0, y_5,  label="MT-DNN", color="red",)
        axs.plot(x_0, y_8,  label="MT-DNN", color="red")
        # axs.plot(x_0, y_6,  label="Tcp")

        axs.scatter(x_0, y_1, color="#6F0000")
        axs.scatter(x_0, y_2, color="#333333")
        axs.scatter(x_0, y_3, color="#999999")
        axs.scatter(x_0, y_4, color="blue")
        # axs.scatter(x_0, y_5, color="red")
        axs.scatter(x_0, y_8, color="red")

        y_ticks = [0.0,  0.1,  0.2, 0.3]
        plt.xticks(x_0, colo_names, rotation=45, fontsize=14)
    else:
        x_1 = [1.5*i - 0.3 for i in range(len(colo_names))]
        x_2 = [1.5*i for i in range(len(colo_names))]
        x_3 = [1.5*i + 0.3 for i in range(len(colo_names))]
        x_4 = [1.5*i + 0.6 for i in range(len(colo_names))]
        y_ticks = [0.0, 0.05, 0.1, 0.15, 0.2]
        axs.bar(
            x_1, y_1, 0.3,
            label="FCFS",
            color="none",
            hatch="////////",
            edgecolor="#CC6600"
        )
        axs.bar(
            x_2, y_2, 0.3,
            label="SJF",
            color="none",
            hatch="////////",
            edgecolor="#006600"
        )
        axs.bar(
            x_3, y_3, 0.3,
            label="EDF",
            color="none",
            hatch="////////",
            edgecolor="#006666"
        )
        axs.bar(
            x_4, y_4, 0.3,
            label="MT-DNN",
            color="none",
            hatch="////////",
            edgecolor="#FF3333"
        )
        plt.xticks(x_2, colo_names, rotation=0, fontsize=14)

    fcfs_improve = (np.array(y_5)-np.array(y_1))/np.array(y_1)
    sjf_improve = (np.array(y_5)-np.array(y_2))/np.array(y_2)
    edf_improve = (np.array(y_5)-np.array(y_3))/np.array(y_3)
    abacus_improve = (np.array(y_8)-np.array(y_4))/np.array(y_4)

    print("FCFS throughout improve:", np.mean(fcfs_improve))
    print("SJF throughout improve:", np.mean(sjf_improve))
    print("EDF throughout improve:", np.mean(edf_improve))
    print("Abacus throughout improve:", np.mean(abacus_improve))
    resnet101 = np.append(abacus_improve[0], abacus_improve[5:9].flatten())
    print("Abacus Resnet101 improve:", np.mean(resnet101))
    print([i for i in range(len(y_4)) if y_4[i] > y_8[i]])

    axs.legend(fontsize=16)
    # axs.yaxis.set_label_coords(-0.09, 0.2)
    # axs.set_ylim(0.12,0.25)
    # axs.yaxis.labelpad = 20
    axs.tick_params(labelsize=14)
    plt.tight_layout()
    axs.set_ylabel("Peek Throughout", loc="center", fontsize=18)
    axs.set_xlabel("Combinations", loc="center", fontsize=18)
    plt.tight_layout()
    plt.savefig("./throughout_with_load.png", bbox_inches="tight")


if __name__ == '__main__':
    result_dir = "/home/onceas/wanna/mt-dnn/results/2080Ti"
    file_names = os.listdir(result_dir)
    throu_dict = {}
    for file in file_names:
        if "2in7-load" not in file:
            continue
        # res_path = os.path.join(result_dir, file, "result.csv")
        res_path = os.path.join(result_dir,  file, "result-throughout.csv")
        # comb = file.split("load")[0].split("-")[1]
        load = file.split("2in7-load")[1]
        if load != "80":
            continue
        throughout, latency_tail = load_single_file(res_path)
        throu_dict[load] = throughout
    comb_dict = {}
    for k, v in throu_dict.items():
        # print(k, v)
        for ty in v:
            if (ty[0] not in comb_dict.keys()):
                comb_dict[ty[0]] = {}
            comb_dict[ty[0]][k] = ty[1]
    print(comb_dict)
    plot_throu_with_load(comb_dict, 80, "plot")

# %%
