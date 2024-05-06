#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import torch.multiprocessing as mp
import datetime
import numpy as np
import itertools
import random
from tqdm import tqdm
import csv
import os
import time
from pkg.option import RunConfig
from pkg.worker import ProfilerWorker

import torch.autograd.profiler as profiler


def gen_model_combinations(models, profiling_combinations=None):
    # id_combinations = [i for i in range(len(models)) for j in range(combination_len)]
    # id_combinations = set(itertools.combinations(id_combinations, combination_len))
    # print(id_combinations)
    # for profiled in done_combinations:
    #     id_combinations.remove(profiled)
    # id_combinations = list(id_combinations)
    # id_combinations = sorted(id_combinations, key=lambda x: (x[0], x[1]))
    model_combinations = []
    for id_comb in profiling_combinations:
        if isinstance(id_comb, int):
            id_comb = [id_comb]
        model_comb = []
        for id in id_comb:
            model_comb.append(models[id])
        model_combinations.append(model_comb)
    print(model_combinations)
    return model_combinations


# new [0,end]
# qos [0,end]
def gen_partition(model_len, if_qos=False, if_new=False):
    start = 0
    end = model_len
    return start, end


# new [0,end]
# qos [0,0]
def gen_partition2(model_len, if_qos=False, if_new=False):
    start = 0
    # resnet50
    if model_len == 18:
        end = model_len
    else:
        end = 0
    return start, end


# new [0,0]
# qos [0,end]
def gen_partition3(model_len, if_qos=False, if_new=False):
    start = 0
    if model_len == 18:
        end = 0
    else:
        end = model_len
    return start, end


# new [random,random]
# qos [0,end]
# qos是整个
# new是随机start-end
def gen_partition4(model_len, if_qos=False, if_new=False):
    if if_qos:
        start = 0
        end = model_len
    else:
        start = random.randrange(0, model_len)
        end = random.randrange(start, model_len + 1)
    return start, end


# new 无
# qos [0,end]  qos是整个
def gen_partition5(model_len, if_qos=False, if_new=False):
    if if_qos:
        start = 0
        end = model_len
    else:
        start = 0
        end = 0
    return start, end


# new 无
# qos [0,end]  qos是整个
def gen_partition6(i, model_len, if_qos=False, if_new=False):
    if i == 0:
        start = 0
        end = model_len
    else:
        if if_new:
            start = 0
            end = random.randrange(0, model_len)
        else:
            start = random.randrange(0, model_len)
            end = model_len
    return start, end


def gen_partition7(model_len, if_qos=False, if_new=False):
    return 0, model_len


def make_record(model_config, raw_record):
    record_max = np.max(raw_record)
    record_min = np.min(raw_record)
    formated_record = np.delete(
        raw_record, np.where((raw_record == record_max) |
                             (raw_record == record_min))
    )
    median = np.median(formated_record)
    mean = np.mean(formated_record)
    var = np.std(formated_record)
    record = [j for sub in model_config for j in sub]
    record.append(median)
    record.append(mean)
    record.append(var)
    return record


def profile(run_config: RunConfig):
    # 阻止调用，直到通信子中所有进程 都调用了该函数后才继续执行
    # 主process+model个子procee
    barrier = mp.Barrier(run_config.total_models + 1)
    profile_data_path = os.path.join(
        run_config.data_path, "profile", run_config.GPU, "2in7")
    os.makedirs(profile_data_path, exist_ok=True)
    for model_combination in gen_model_combinations(
        run_config.models_name,
        run_config.profiling_combinations,
    ):
        print(model_combination)

        profile_filename = model_combination[0]
        for model_name in model_combination[1:]:
            profile_filename = profile_filename + "_" + model_name
        profile_filename += ".csv"

        profile_file = open(os.path.join(
            profile_data_path, profile_filename), "w+")
        wr = csv.writer(profile_file, dialect="excel")
        # input features
        profile_head = [
            "model",
            "start",
            "end",
            "bs",
            "seq_len",
        ] * run_config.total_models + [
            "median",
            "mean",
            "var",
        ]
        wr.writerow(profile_head)

        worker_list = []
        # 每个pipe对应一个model，有多少个model就有多少个pipe
        # 一共fork model个process
        for worker_id, model_name in enumerate(model_combination):
            pipe_parent, pipe_child = mp.Pipe()
            # 接收action，进行model实际的推理过程
            # model_name, action, start, end, bs, seq_len = self._pipe.recv()

            # TODO
            partition = "100"
            # if model_name == "resnet50":
            #     partition = "10"
            # else:
            #     continue
            model_worker = ProfilerWorker(
                run_config,
                model_name,
                run_config.supported_batchsize,
                run_config.supported_seqlen,
                pipe_child,
                barrier,
                worker_id,
                partition,
            )
            model_worker.start()
            # pipe_parent发送参数，model_worker根据参数运行
            worker_list.append((model_worker, pipe_parent))
        barrier.wait()

        # 从run_config.supported_batchsize中挑选[32,32]等
        for bs_it in itertools.product(
            run_config.supported_batchsize, repeat=run_config.total_models
        ):
            model_ids = [i for i in range(run_config.total_models)]
            profiled_config = set()
            # 对每种batch_size组合，run_config.total_test默认200，即产生200种组合
            for test_i in range(run_config.total_test):
                model_config = []
                # qos和new的个数都是 1~model_num个数
                qos_query_cnt = random.randrange(
                    1, run_config.total_models + 1)
                new_query_cnt = random.randrange(
                    1, run_config.total_models + 1)
                # qos有一组，new query有一组，各包含几个model
                qos_ids = random.sample(model_ids, qos_query_cnt)
                new_ids = random.sample(model_ids, new_query_cnt)

                # TODO fix 整个+无新来
                # while True:
                #     qos_ids = random.sample(model_ids, 1)
                #     if model_combination[qos_ids[0]] == 'resnet50':
                #         break
                # new_ids = []

                # TODO fix 整个+无新来
                qos_ids = model_ids

                print("qos:{} new:{}".format(qos_ids, new_ids))
                # while True:
                #     new_ids = random.sample(model_ids, 1)
                #     if model_combination[new_ids[0]] != 'resnet50':
                #         break
                for i in range(run_config.total_models):
                    # 参照论文figure9思想
                    # qos 从以前的random start开始 [start,model_len]
                    # new 到一个random end结束     [0,end]
                    # 如果new 和 qos 有重叠：取整个model  [0,model_len]
                    # 不是new也不是qos random start和end [start,end]
                    # start, end = gen_partition4(
                    #     run_config.models_len[model_combination[i]],
                    #     True if i in qos_ids else False,
                    #     True if i in new_ids else False,
                    # )
                    start, end = gen_partition5(
                        run_config.models_len[model_combination[i]],
                        True if i in qos_ids else False,
                        True if i in new_ids else False,
                    )
                    # start, end = gen_partition6(
                    #     i, run_config.models_len[model_combination[i]],
                    #     True if i in qos_ids else False,
                    #     True if i in new_ids else False,
                    # )
                    # start, end = gen_partition7(
                    #     run_config.models_len[model_combination[i]],
                    #     True if i in qos_ids else False,
                    #     True if i in new_ids else False,
                    # )
                    seq_len = (
                        random.choice(run_config.supported_seqlen)
                        if model_combination[i] == "bert"
                        else 0
                    )
                    model_config.append(
                        [model_combination[i], start, end, bs_it[i], seq_len]
                    )
                # 去重
                pendding_profile_config = tuple(tuple(i) for i in model_config)
                if pendding_profile_config in profiled_config:
                    if run_config.total_models == 1:
                        print(
                            "Profiled model config: {}, {}, {}, {}, {}".format(
                                model_config[0][0],
                                model_config[0][1],
                                model_config[0][2],
                                model_config[0][3],
                                model_config[0][4],
                            )
                        )
                    else:
                        print(
                            "Profiled model config: {}, {}, {}, {}, {},{}, {}, {}, {}, {}".format(
                                model_config[0][0],
                                model_config[0][1],
                                model_config[0][2],
                                model_config[0][3],
                                model_config[0][4],
                                model_config[1][0],
                                model_config[1][1],
                                model_config[1][2],
                                model_config[1][3],
                                model_config[1][4],
                            )
                        )
                else:
                    profiled_config.add(pendding_profile_config)
                    for i in range(run_config.total_models):
                        _, model_pipe = worker_list[i]
                        model_pipe.send(
                            (
                                model_config[i][0],
                                "prepare",
                                model_config[i][1],
                                model_config[i][2],
                                model_config[i][3],
                                model_config[i][4],
                            )
                        )
                    barrier.wait()
                    record = []
                    with tqdm(range(run_config.test_loop)) as t:
                        for loop_i in t:
                            start_time = datetime.datetime.now()
                            for i in range(run_config.total_models):
                                _, model_pipe = worker_list[i]
                                model_pipe.send(
                                    (
                                        model_config[i][0],
                                        "forward",
                                        model_config[i][1],
                                        model_config[i][2],
                                        model_config[i][3],
                                        model_config[i][4],
                                    )
                                )
                            # barrier.wait()
                            # start_time = datetime.datetime.now()
                            barrier.wait()
                            elapsed_time_us = (
                                datetime.datetime.now() - start_time
                            ).microseconds
                            # 进度条log
                            t.set_postfix(elapsed=elapsed_time_us)
                            # 手动更新进度条
                            t.update(1)
                            record.append(elapsed_time_us)

                    profile_record = make_record(model_config, record)
                    wr.writerow(profile_record)
                    profile_file.flush()
                    # time.sleep(10)
        for i in range(run_config.total_models):
            _, model_pipe = worker_list[i]
            model_pipe.send(("none", "terminate", -1, -1, -1, -1))

        for worker, _ in worker_list:
            worker.join()
