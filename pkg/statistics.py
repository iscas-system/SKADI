#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import glob
import pandas as pd
import matplotlib.ticker as mtick
import os
import random

# project_path = "/home/onceas/wanna/mt-dnn/results/workloads/12/cluster-0.77/"
project_path = "/home/onceas/wanna/mt-dnn/results/cluster-0.80/"
abacus_src_path = os.path.join(project_path, "Abacus")
abacus_raw_file = os.path.join(
    project_path, "Abacus_raw.csv")
abacus_clean_file = os.path.join(
    project_path, "Abacus_clean.csv")

clock_src_path = project_path + "Clock"
clock_raw_file = os.path.join(
    project_path, "Clock_raw.csv")
clock_clean_file = os.path.join(
    project_path, "Clock_clean.csv")

loads = [
    8357,
    9005,
    8908,
    8771,
    8624,
    8393,
    10345,
    9490,
    9388,
    9498,
    9108,
    9337,
    10606,
    10682,
    10338,
    10069,
    9691,
    8936,
    8723,
    8951,
    8796,
    9293,
    9207,
    8999,
    8779,
    8731,
    9265,
    9325,
    9206,
    8913,
    8840,
    8752,
    8958,

    # 8712,
    # 8974,
    # 8430,
    # 9704,
    # 9170,
    # 8902,
    # 8954,
    # 8668,
    # 8986,
    # 8846,
    # 8640,
    # 8437,
    # 8944,
    # 9248,
    # 8851,
    # 8725,
    # 8645,
    # 8627,
    # 8929,
    # 8809,
    # 8850,
    # 8823,
    # 8873,
    # 9179,
    # 8522,
    # 8737,
    # 8851,
    # 8689,
    # 8538,
    # 8702,
    # 8683,
    # 8726,
    # 8780,
    # 10904,
    # 9764,
    # 9295,
    # 9504,
    # 9509,
    # 9663,
    # 10498,
    # 10480,
    # 10450,
    # 10264,
    # 10107,
    # 9409,
    # 8987,
    # 8920,
    # 8719,
    # 8863,
    # 8931,
    # 9015,
    # 9117,
    # 8812,
    # 9545,
    # 9038,
    # 8698,
    # 9091,
    # 8579,
    # 9014,
    # 8794,
    # 8621,
    # 8876,
    # 8839,
    # 9782,
    # 9011,
    # 8772,
    # 9180,
    # 8875,
    # 9124,
    # 8721,
    # 8875,
    # 8732,
    # 8770,
    # 9435,
    # 8944,
    # 8914,
    # 8793,
    # 8701,
    # 9013,
    # 8768,
    # 8887,
    # 8621,
    # 9190,
    # 9231,
    # 9021,
    # 8781,
    # 8905,
]


def merge_abacus(src_path, dst_file):
    file_cnt = 0
    for filename in glob.glob(src_path + "/**/*.csv", recursive=True):
        if "resnet50resnet101" not in filename:
            continue
        # print("file: {} loaded".format(filename))
        df = pd.read_csv(filename, header=0)
        if file_cnt == 0:
            df.to_csv(dst_file, index=False)
            file_cnt += 1
        else:
            df.to_csv(dst_file, index=False, header=False, mode="a+")


def merge_clock(src_path, dst_file):
    file_cnt = 0
    for filename in glob.glob(src_path + "/*.csv", recursive=True):
        print("file: {} loaded".format(filename))
        df = pd.read_csv(filename, header=0)
        if file_cnt == 0:
            df.to_csv(dst_file, index=False)
            file_cnt += 1
        else:
            df.to_csv(dst_file, index=False, header=False, mode="a+")


def abacus_preprocess(src_name, dst_name):
    data = pd.read_csv(src_name, header=0)
    # print(len(data))
    data_resorted = data.sort_values("query_id")
    data_removed = data_resorted[data_resorted.query_id != 0]
    data_removed = data_removed.reset_index(drop=True)
    # start_index = 0
    # total_query = len(data_removed)
    # total_load = sum(loads)
    # for i in range(len(abacus_load)):
    #     # print("load id: {}".format(i))
    #     end_index = int(start_index + total_query * loads[i] / total_load)
    #     # print("end_index: {}".format(end_index))
    #     if i == len(abacus_load)-1:
    #         end_index = total_query
    #     data_removed.loc[start_index:end_index, "load_id"] = i
    #     start_index = end_index
    data_removed.to_csv(dst_name, index=False)


def clock_preprocess(src_name, dst_name):
    data = pd.read_csv(src_name, header=0)
    data_resorted = data.sort_values("query_id")
    data_removed = data_resorted[data_resorted.query_id != 0]
    data_removed = data_removed.reset_index(drop=True)
    data_removed.to_csv(dst_name, index=False)


def get_abacus_data(src_df):
    src_df = src_df[src_df["latency"] != -1]
    # src_df = src_df[src_df["latency"] < 105]
    src_df = src_df[src_df["latency"] < 100]
    return src_df


def get_load(src_df):
    load_df = src_df[src_df["latency"] != -1]
    return len(load_df) / 60


def get_clock_data(src_df):
    src_df = src_df[src_df["latency"] != -1]
    # src_df = src_df[src_df["latency"] < 103]
    src_df = src_df[src_df["latency"] < 100]
    return src_df


def get_abacus_latency(file_name):
    data = pd.read_csv(file_name, header=0)
    data_plot1 = []
    data_plot2 = []
    for i in range(30):
        raw_df = data[data["load_id"] == i]
        raw_df = raw_df[raw_df["model_id"] == 0]
        cur_load = get_load(raw_df)
        df = get_abacus_data(raw_df)
        cur_lat_99 = df.latency.quantile(0.99)
        cur_lat_mean = df.latency.mean()
        sla_violation = (len(raw_df) - len(df))/len(raw_df)
        # print(
        #     "load: {},99%-ile: {}, mean: {}, load: {}".format(
        #         i, cur_lat_99, cur_lat_mean, cur_load)
        # )
        if i == 0:
            data_plot1 = np.array(
                [[cur_lat_99, cur_lat_mean, cur_load, sla_violation]])
        else:
            data_plot1 = np.append(
                data_plot1, np.array([[cur_lat_99, cur_lat_mean, cur_load, sla_violation]]), axis=0
            )

    for i in range(30):
        raw_df = data[data["load_id"] == i]
        raw_df = raw_df[raw_df["model_id"] == 1]
        cur_load = get_load(raw_df)
        df = get_abacus_data(raw_df)
        cur_lat_99 = df.latency.quantile(0.99)
        cur_lat_mean = df.latency.mean()
        sla_violation = (len(raw_df) - len(df))/len(raw_df)
        # print(
        #     "load: {},99%-ile: {}, mean: {}, load: {}".format(
        #         i, cur_lat_99, cur_lat_mean, cur_load)
        # )
        if i == 0:
            data_plot2 = np.array(
                [[cur_lat_99, cur_lat_mean, cur_load, sla_violation]])
        else:
            data_plot2 = np.append(
                data_plot2, np.array([[cur_lat_99, cur_lat_mean, cur_load, sla_violation]]), axis=0
            )
    return data_plot1, data_plot2


def get_clock_latency(file_name):
    data = pd.read_csv(file_name, header=0)
    data_plot = []
    for i in range(30):
        # if i == 0:
        #     raw_df = data[data["load_id"] == 1]
        # elif i == 1:
        #     raw_df = data[data["load_id"] == 2]
        # else:
        raw_df = data[data["load_id"] == i]

        df = get_clock_data(raw_df)
        sla_violation = (len(raw_df) - len(df))/len(raw_df)
        cur_load = get_load(df)
        cur_lat_99 = df.latency.quantile(0.99)
        cur_lat_mean = df.latency.mean()
        # print(
        #     "load: {},99%-ile: {}, mean: {}, load: {}".format(
        #         i, cur_lat_99, cur_lat_mean, cur_load)
        # )
        if i == 0:
            data_plot = np.array(
                [[cur_lat_99, cur_lat_mean, cur_load, sla_violation]])
        else:
            data_plot = np.append(
                data_plot, np.array([[cur_lat_99, cur_lat_mean, cur_load, sla_violation]]), axis=0
            )
    return data_plot


if __name__ == "__main__":

    # draw_type = "profile"
    draw_type = "paper"

    merge_abacus(src_path=abacus_src_path, dst_file=abacus_raw_file)
    merge_clock(src_path=clock_src_path, dst_file=clock_raw_file)

    abacus_preprocess(abacus_raw_file, abacus_clean_file)
    clock_preprocess(clock_raw_file, clock_clean_file)

    abacus_data = get_abacus_latency(abacus_clean_file)
    # print(abacus_data)
    clock_data = get_clock_latency(clock_clean_file)
    # print(clock_data)

    abacus_99 = abacus_data[:30, 0]
    abacus_mean = abacus_data[:30, 1]
    abacus_load = abacus_data[:30, 2]
    abacus_vio = abacus_data[:30, 3]

    clock_99 = clock_data[:30, 0]
    clock_mean = clock_data[:30, 1]
    clock_load = clock_data[:30, 2]
    clock_vio = clock_data[:30, 3]

    x = [i for i in range(len(abacus_load))]
    shuffle_x = [i for i in range(len(abacus_load))]
    random.shuffle(shuffle_x)

    print(len(x))
    print(abacus_vio)
    print(clock_vio)

    load_improve = sum(abacus_load) / sum(clock_load) - 1
    print("throughout improve:", load_improve)
    vio_items = np.array(clock_vio-abacus_vio)/np.array(clock_vio)
    print("sla violation reduce:", vio_items.mean())

    # mpl.use("Agg")
    mpl.rcParams["font.family"] = "Times New Roman"

    if draw_type == "profile":
        fig, axs = plt.subplots(4, 1, figsize=(10, 9))
    elif draw_type == "paper":
        fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    fig.tight_layout(pad=0.1)
    # y_loads = [item/16/7 for item in loads[:len(abacus_load)]]
    y_loads = [loads[item]/16/6 for item in shuffle_x]
    abacus_load_shuffle = [abacus_load[item] for item in shuffle_x]
    clock_load_shuffle = [clock_load[item] for item in shuffle_x]
    print(y_loads)
    axs[0].plot(x, y_loads, label="Request load", color="#333333")
    axs[0].plot(x, abacus_load_shuffle, label="MT-DNN", color="red")
    axs[0].plot(x, clock_load_shuffle, label="Clock", color="blue")
    # axs[0].set_size_inches(10,3)

    abacus_vio_shuffle = [abacus_vio[item] for item in shuffle_x]
    clock_vio_shuffle = [clock_vio[item] for item in shuffle_x]
    axs[1].plot(x, abacus_vio_shuffle, color="red")
    axs[1].plot(x, clock_vio_shuffle, color="blue")
    # axs[1].set_size_inches(10,3)

    if draw_type == "profile":
        axs[2].plot(x, abacus_99, color="red")
        axs[2].plot(x, clock_99, color="blue")

        axs[3].plot(x, abacus_mean, color="red")
        axs[3].plot(x, clock_mean, color="blue")

    # axs[0].set_xlim(0, len(abacus_load)-1)
    # axs[0].set_ylim(7000, 11000)
    axs[0].set_ylabel("Tput\n(r/s)", fontsize=16)
    # axs[0].set_yticks([7000, 8000, 9000, 10000, 11000])
    # axs[0].set_yticklabels(["7k", "8k", "9k", "10k", "11k"], fontsize=16)
    # axs[0].set_xticklabels([])

    axs[1].set_ylabel("SLA Violation", fontsize=16)
    axs[1].set_xlabel("Timeline (minutes)", fontsize=16)
    axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
    axs[1].set_yticks([0.05, 0.1, 0.15, 0.2])

    if draw_type == "profile":
        # axs[2].set_xlim(0, len(abacus_load)-1)
        # axs[2].set_ylim(85, 102)
        axs[2].set_ylabel("99%-ile Lat\n(ms)", fontsize=16)
        # axs[2].set_yticks([85, 90, 95, 100])
        # axs[2].set_yticklabels(["85", "90", "95", "100"], fontsize=16)
        # axs[2].set_xticklabels([])

        # axs[3].set_xlim(0, len(abacus_load)-1)
        # axs[3].set_ylim(20, 50)
        axs[3].set_ylabel("Avg Lat\n(ms)", fontsize=16)
        axs[3].set_xlabel("Timeline (minutes)", fontsize=16)
        # axs[3].set_yticks([20, 30, 40, 50])
        # axs[3].set_yticklabels(["20", "30", "40", "50"], fontsize=16)
        # axs[3].set_xticks([0, 20, 40, 60, 80, 100, 119])
        # axs[3].set_xticklabels(
        #     ["0", "20", "40", "60", "80", "100", "120"], fontsize=16)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=3, fontsize=16, bbox_to_anchor=(0.8, 1.10), frameon=False
    )
    # plt.tight_layout()
    # plt.savefig(project_path + "figure/large_scale.pdf", bbox_inches="tight")
    # plt.show()


# %%
