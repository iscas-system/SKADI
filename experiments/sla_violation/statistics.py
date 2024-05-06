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


def get_latency(data):
    n = data.shape[0]
    latency_data = []
    for i in range(n):
        line = data[i]
        latency = [line[0], [float(line[3]), float(line[6]),
                   float(line[9]), float(line[12]), float(line[15]), float(line[18]), float(line[21]), float(line[24])]]
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
    return get_latency(data), get_latency_tail(data)


def plot_vio_with_load(sla_vio_dict: dict):
    mpl.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(24, 8))
    # fig = plt.figure(figsize=(18, 8))
    idx = 1
    for items in sla_vio_dict.items():
        # print(items[0])
        if not ("Res50+Res152" == items[0] or "Res152+VGG19" == items[0]):
            continue
        # print("comb:", items[0])
        # print(items[1])
        sorted_items = sorted(items[1].items(), key=lambda x: int(x[0]))
        print(sorted_items)
        axs = fig.add_subplot(1, 2, idx)
        idx += 1
        x_1 = [i for i in range(1, len(sorted_items)+1)]
        colo_names = [10*(i-1) for i in x_1]
        colo_names[0] = 5
        y_1 = [it[1][0] for it in sorted_items]
        y_2 = [it[1][1] for it in sorted_items]
        y_3 = [it[1][2] for it in sorted_items]
        y_4 = [it[1][3] for it in sorted_items]
        y_5 = [it[1][4] for it in sorted_items]
        y_6 = [it[1][5] for it in sorted_items]
        y_7 = [it[1][6] for it in sorted_items]

        axs.plot(x_1, y_1, label="FCFS", color="#6F0000")
        axs.plot(x_1, y_2, label="SJF", color="#333333")
        axs.plot(x_1, y_3, label="EDF", color="#999999")
        # axs.plot(x_1, y_4, color="red", label="Abacus")
        axs.plot(x_1, y_4,  label="MT-DNN", color="red")
        # axs.plot(x_1, y_5,  label="MT-DNN", color="red")
        # axs.plot(x_1, y_6,  label="Tcp")
        # axs.plot(x_1, y_7,  label="Linear")

        # axs.plot(x_1, y_1,color="teal")
        axs.set_ylim(0, 1.1)
        axs.set_xlim(0, 18)
        axs.legend(fontsize=16)

        # axs.yaxis.set_label_coords(-0.09, 0.2)
        axs.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        # axs.set_ylim(0.12,0.25)
        # axs.yaxis.labelpad = 20
        axs.tick_params(labelsize=16)
        plt.xticks(x_1, colo_names, rotation=45, fontsize=14)
        plt.tight_layout()
        axs.set_ylabel(items[0]+" Qos Violation", loc="center", fontsize=18)
        axs.set_xlabel("Request load", loc="center", fontsize=18)
    plt.tight_layout()
    plt.savefig("./qos_violation_ratio_with_load.png", bbox_inches="tight")


def plot_latency_tail_with_load(latency_dict: dict):
    mpl.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(18, 8))
    idx = 1
    for items in latency_dict.items():
        # print("comb:", items[0])
        # print(items[1])
        sorted_items = sorted(items[1].items(), key=lambda x: int(x[0]))
        print(sorted_items)
        axs = fig.add_subplot(1, 2, idx)
        idx += 1
        x_1 = [i for i in range(1, len(sorted_items)+1)]
        colo_names = [10*(i-1) for i in x_1]
        colo_names[0] = 5
        y_1 = [it[1][0] for it in sorted_items]
        y_2 = [it[1][1] for it in sorted_items]
        y_3 = [it[1][2] for it in sorted_items]
        y_4 = [it[1][3] for it in sorted_items]
        y_5 = [it[1][4] for it in sorted_items]
        y_6 = [it[1][5] for it in sorted_items]

        axs.plot(x_1, y_1, label="FCFS")
        axs.plot(x_1, y_2, label="SJF")
        axs.plot(x_1, y_3, label="EDF")
        axs.plot(x_1, y_4, color="red", label="Abacus")
        axs.plot(x_1, y_5,  label="MT-DNN")
        axs.plot(x_1, y_6,  label="Tcp")

        # axs.plot(x_1, y_1,color="teal")
        axs.set_ylim(0, 1.1)
        axs.set_xlim(0, 18)
        axs.legend(fontsize=16)

        # axs.yaxis.set_label_coords(-0.09, 0.2)
        # axs.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        # axs.set_ylim(0.12,0.25)
        # axs.yaxis.labelpad = 20
        axs.tick_params(labelsize=16)
        plt.xticks(x_1, colo_names, rotation=0, fontsize=14)
        plt.tight_layout()
        axs.set_ylabel(items[0]+" 99% Latency Tail", loc="center", fontsize=18)
        axs.set_xlabel("Request load", loc="center", fontsize=18)
    plt.savefig("./latency_tail_with_load.png", bbox_inches="tight")


def plot_vio_with_fixed_load(latency_dict: dict, fixed_load: int, figure_type: str):
    mpl.rcParams["font.family"] = "Times New Roman"
    fig = plt.figure(figsize=(12, 8))
    idx = 1
    fixed_load_dict = {}
    for items in latency_dict.items():
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
    y_1 = [fixed_load_dict[it][0] for it in colo_names]
    y_2 = [fixed_load_dict[it][1] for it in colo_names]
    y_3 = [fixed_load_dict[it][2] for it in colo_names]
    y_4 = [fixed_load_dict[it][3] for it in colo_names]
    y_5 = [fixed_load_dict[it][4] for it in colo_names]
    y_6 = [fixed_load_dict[it][5] for it in colo_names]
    y_8 = [fixed_load_dict[it][7] for it in colo_names]

    print(y_4)
    print(y_8)
    print([i for i in range(len(y_4)) if y_4[i] < y_8[i]])
    print(np.array(y_4).mean(), np.array(y_8).mean())
    if figure_type == "plot":
        axs.plot(x_0, y_1, label="FCFS", color="#6F0000")
        axs.plot(x_0, y_2, label="SJF", color="#333333")
        axs.plot(x_0, y_3, label="EDF", color="#999999")
        axs.plot(x_0, y_4,  label="Abacus", color="blue")
        # axs.plot(x_0, y_5,  label="MT-DNN", color="red")
        axs.plot(x_0, y_8,  label="MT-DNN", color="red")
        # axs.plot(x_0, y_5,  label="MT-DNN")
        # axs.plot(x_0, y_6,  label="Tcp")

        axs.scatter(x_0, y_1, color="#6F0000")
        axs.scatter(x_0, y_2, color="#333333")
        axs.scatter(x_0, y_3, color="#999999")
        axs.scatter(x_0, y_4, color="blue")
        # axs.scatter(x_0, y_5, color="red")
        axs.scatter(x_0, y_8, color="red")

        y_ticks = [0.0,  0.1,  0.2, 0.3, 0.4, 0.5]
        # y_ticks = [0.0,  0.1,  0.2, 0.3]
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

    fcfs_reduce = (np.array(y_1)-np.array(y_8))/np.array(y_1)
    sjf_reduce = (np.array(y_2)-np.array(y_8))/np.array(y_2)
    edf_reduce = (np.array(y_3)-np.array(y_8))/np.array(y_3)
    # abacus_reduce = (np.array(y_4)-np.array(y_5))/np.array(y_4)
    mtdnn3_reduce = (np.array(y_4)-np.array(y_8))/np.array(y_4)
    mtdnn_reduce = (np.array(y_5)-np.array(y_8))/np.array(y_5)

    print("FCFS reduce:", np.mean(fcfs_reduce))
    print("SJF reduce:", np.mean(sjf_reduce))
    print("EDF reduce:", np.mean(edf_reduce))
    # print("MT-DNN to Abacus reduce:", np.mean(abacus_reduce))
    resnet101 = np.append(mtdnn3_reduce[0], mtdnn3_reduce[5:9].flatten())
    print("MT-DNN to Abacus Resnet101 reduce:", np.mean(resnet101))

    print("MT-DNN3 to Abacus reduce:", np.mean(mtdnn3_reduce))
    # print("MT-DNN3 to MT-DNN reduce:", np.mean(mtdnn_reduce))

    axs.yaxis.set_major_formatter(mtick.PercentFormatter(1))

    # y_ticks = [0.0, 0.1, 0.2]
    # axs.plot(x_1, y_1,color="teal")
    # axs.set_ylim(0, 1.1)
    # axs.set_xlim(0, 18)
    axs.legend(fontsize=16)

    # axs.yaxis.set_label_coords(-0.09, 0.2)
    # axs.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    # axs.set_ylim(0.12,0.25)
    # axs.yaxis.labelpad = 20
    axs.tick_params(labelsize=16)
    # plt.xticks(x_2, colo_names, rotation=0, fontsize=14)
    plt.yticks(y_ticks, fontsize=14)
    plt.tight_layout()
    axs.set_ylabel("SLA Violation Rate",
                   loc="center", fontsize=18)
    axs.set_xlabel("Combinations", loc="center", fontsize=18)
    plt.savefig("./violation_fixed_load.png", bbox_inches="tight")


if __name__ == '__main__':
    result_dir = "/home/onceas/wanna/mt-dnn/results/2080Ti"
    file_names = os.listdir(result_dir)
    sla_vio_dict = {}
    latency_tail_dict = {}
    for file in file_names:
        if "2in7-load" not in file:
            continue
        load = file.split("2in7-load")[1]
        # if load != "60":
        #     continue
        # res_path = os.path.join(result_dir, file, "result.csv")
        res_path = os.path.join(result_dir,  file, "result.csv")
        # comb = file.split("load")[0].split("-")[1]

        sla_vio, latency_tail = load_single_file(res_path)
        sla_vio_dict[load] = sla_vio
        latency_tail_dict[load] = latency_tail
    comb_dict = {}
    latency_dict = {}
    for k, v in sla_vio_dict.items():
        # print(k, v)
        for ty in v:
            if (ty[0] not in comb_dict.keys()):
                comb_dict[ty[0]] = {}
            comb_dict[ty[0]][k] = ty[1]
    for k, v in latency_tail_dict.items():
        for ty in v:
            if (ty[0] not in latency_dict.keys()):
                latency_dict[ty[0]] = {}
            latency_dict[ty[0]][k] = ty[1]

    print(comb_dict)
    # print(latency_dict)
    # plot_vio_with_load(comb_dict)
    # plot_latency_tail_with_load(latency_dict)
    plot_vio_with_fixed_load(comb_dict, 50, "plot")

# %%
