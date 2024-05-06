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

from pkg.option import RunConfig
from pkg.worker import ProfilerWorker
from pkg.utils import gen_model_combinations, gen_partition, make_record


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
            model_worker = ProfilerWorker(
                run_config,
                model_name,
                run_config.supported_batchsize,
                run_config.supported_seqlen,
                pipe_child,
                barrier,
                worker_id,
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
                qos_query_cnt = random.randrange(
                    1, run_config.total_models + 1)
                new_query_cnt = random.randrange(
                    1, run_config.total_models + 1)
                # qos有一组，new query有一组，各包含几个model
                qos_ids = random.sample(model_ids, qos_query_cnt)
                new_ids = random.sample(model_ids, new_query_cnt)
                for i in range(run_config.total_models):
                    # 参照论文figure9思想
                    # qos 从以前的random start开始 [start,model_len]
                    # new 到一个random end结束     [0,end]
                    # 如果new 和 qos 有重叠：取整个model  [0,model_len]
                    # 不是new也不是qos random start和end [start,end]
                    start, end = gen_partition(
                        run_config.models_len[model_combination[i]],
                        True if i in qos_ids else False,
                        True if i in new_ids else False,
                    )
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
        for i in range(run_config.total_models):
            _, model_pipe = worker_list[i]
            model_pipe.send(("none", "terminate", -1, -1, -1, -1))

        for worker, _ in worker_list:
            worker.join()
