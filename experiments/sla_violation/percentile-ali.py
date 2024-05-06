#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# %%

import numpy as np
import os
import pandas as pd
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

pai_machine_metric_path = "/home/onceas/wanna/pai_machine_metric.csv"
pai_machine_spec_path = "/home/onceas/wanna/pai_machine_spec.csv"
pai_sensor_path = "/home/onceas/wanna/pai_sensor_table.csv"
pai_sensor_raw_path = "/home/onceas/wanna/pai_sensor_table_raw.csv"
pai_group_path = "/home/onceas/wanna/pai_group_tag_table.csv"


def get_gpu_util(data):
    # pai_machine_metric_path
    n = len(data)
    gpu_util_dict = {}
    for i in range(n):
        # line = data.iloc[i]
        line = data[i]
        if np.isnan(line[1]) or line[1] == "":
            continue
        gpu_util_dict[line[0]] = float(line[1])
    print(len(gpu_util_dict))
    return gpu_util_dict


def get_gpu_num(data):
    # pai_machine_spec_path
    n = len(data)
    gpu_num_dict = {}
    for i in range(n):
        line = data.iloc[i]
        # if line[2] == "0" or line[1] == "MISC":
        #     continue
        # {'P100', 'V100', 'MISC', 'T4', 'CPU', 'V100M32'}
        if line[1] == "CPU" or line[4] == 0:
            continue
        # print(line)
        if line[0] in gpu_num_dict.keys():
            print(line[0])
        gpu_num_dict[line[0]] = int(line[4])
    print(len(gpu_num_dict))
    return gpu_num_dict


def get_sensor_util(data):
    n = len(data)
    gpu_util_dict = []
    # print(n) # 9774
    for i in range(n):
        line = data.iloc[i]
        # print(line)
        # if line[2] == "0" or line[1] == "MISC":
        #     continue
        # if line[3] not in instance_type.keys():
        #     continue
        # gpu_util_dict[key] = float(line[7])
        gpu_util_dict.append(float(line[7]))
    return gpu_util_dict


def get_group_util(data):
    n = len(data)
    gpu_util_dict = {}
    for i in range(n):
        line = data.iloc[i]
        gpu_util_dict[line[0]] = line[4]
    return gpu_util_dict


def load_single_file(filepath):
    if "sensor" in filepath:
        data = pd.read_csv(filepath, usecols=[0, 1, 3, 7], header=None)
        data = data[data[1] == 'evaluator']
        return get_sensor_util(data)
    elif "tag" in filepath:
        data = pd.read_csv(filepath, usecols=[0, 4], header=None)
        data = data[data[4] == 'ctr']
        return get_group_util(data)
    elif "metric" in filepath:
        print(1)
        data = pd.read_csv(filepath, usecols=[1, 7], header=None)
        data = data.values.tolist()
        return get_gpu_util(data)
    elif "spec" in filepath:
        print(2)
        data = pd.read_csv(filepath, usecols=[0, 1, 4], header=None)
        # data = data.values.tolist()
        return get_gpu_num(data)


def data_preprocess1():
    import csv
    # print("--------------{}--------------".format(file_names[i]))
    gpu_utilization = load_single_file(pai_machine_metric_path)
    gpu_num = load_single_file(pai_machine_spec_path)
    avg_util = np.array([])
    for key in gpu_utilization.keys():
        if key not in gpu_num.keys():
            continue
        # print(gpu_utilization[key], gpu_num[key])
        util = gpu_utilization[key]/gpu_num[key]

        if util > 100:
            # print(key, gpu_utilization[key], gpu_num[key])
            util = 100
        avg_util = np.append(avg_util, util)
    print(len(avg_util))
    lens = len(avg_util)
    print(len(avg_util[avg_util > 25]))
    print("avg_util max:{}".format(np.max(avg_util)))
    print("avg_util mean:{}".format(np.mean(avg_util)))
    # print("avg_util 80%:{}".format(np.percentile(avg_util, 99)))

    print("ratio mionr 40%:{}".format(len(avg_util[avg_util < 40])/lens))

    return avg_util


def data_preprocess2():
    import csv
    # print("--------------{}--------------".format(file_names[i]))
    gpu_utilization = load_single_file(pai_sensor_path)
    gpu_utilization = np.array(gpu_utilization)
    print("ratio mionr 0.042:{}".format(
        len(gpu_utilization[gpu_utilization < 10])/len(gpu_utilization)))
    # gpu_utilization = [gpu_utilization[k] for k in gpu_utilization.keys()]
    return gpu_utilization


def plot_cdf(gpu_utilization):
    if isinstance(gpu_utilization, dict):
        gpu_utilization = [gpu_utilization[k] for k in gpu_utilization.keys()]
    gpu_utilization = np.array(gpu_utilization)/100
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    # ax.set_xlabel("# GPUs", fontsize=20)
    ax.set_xlabel("# GPUs", fontsize=20)
    ax.set_ylabel("CDF", fontsize=20)
    ax.tick_params("x", labelsize=18)
    ax.tick_params("y", labelsize=18)

    sns.ecdfplot(data=gpu_utilization)

    plt.legend(fontsize=16, ncol=2, loc=(0.5, 0), frameon=False)
    # plt.savefig("../figure/unpredictable_latency.png", bbox_inches="tight")
    plt.show()


def data_preprocess_group():
    import csv
    # print("--------------{}--------------".format(file_names[i]))
    instance_type = load_single_file(pai_group_path)
    return instance_type


instance_type = data_preprocess_group()


if __name__ == "__main__":
    # 2080Ti qos
    gpu_utilization = data_preprocess2()
    plot_cdf(gpu_utilization)

    # gpu_utilization = data_preprocess1()
    # plot_cdf(gpu_utilization)


# %%
