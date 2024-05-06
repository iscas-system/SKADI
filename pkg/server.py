#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import logging
import time
import numpy as np
import random
import torch
import csv
from concurrent import futures
import grpc
import json

import torch.multiprocessing as mp
from torch.multiprocessing import Process

import pkg.service_pb2 as service_pb2
import pkg.service_pb2_grpc as service_pb2_grpc
from pkg.option import RunConfig
from pkg.worker import ServerWorker
from pkg.modeling.predictor.mlp import MLPregression
from pkg.utils import Query


class ClockServer(service_pb2_grpc.DNNServerServicer):
    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._node_id = run_config.node_id
        self._warmup_barrier = mp.Barrier(run_config.total_models + 1)
        self._barrier = [mp.Barrier(2)]
        self._stop_event = mp.Event()
        self._workers = {}
        self._queues = {}
        self._pipes = {}
        self._qos_target = run_config.qos_target
        random.seed(0)
        log_path = "results/cluster/" + self._run_config.policy
        log_dir = os.path.join(self._run_config.path, log_path)
        os.makedirs(log_dir, exist_ok=True)
        self._serve_combination = run_config.serve_combination
        result_fname = ""
        for model_id in self._serve_combination:
            result_fname += run_config.models_name[model_id]
        result_fname += ".csv"
        self._result_path = os.path.join(log_dir, result_fname)
        self._result_file = open(self._result_path, "w+")
        self._wr = csv.writer(self._result_file, dialect="excel")
        result_header = ["query_id", "model_id",
                         "bs", "seq_len", "load_id", "latency"]
        self._wr.writerow(result_header)
        self._result_file.flush()
        self.start_up()

    def start_up(self):
        logging.info("Starting Clockworker Workers")
        for worker_id, model_id in enumerate(self._run_config.serve_combination):
            model_name = self._run_config.models_name[model_id]
            pipe_parent, pipe_child = mp.Pipe()
            model_worker = ServerWorker(
                self._run_config,
                model_name,
                self._run_config.supported_batchsize,
                self._run_config.supported_seqlen,
                pipe_child,
                self._barrier,
                self._warmup_barrier,
                worker_id,
            )
            model_worker.start()
            self._workers[model_id] = model_worker
            self._pipes[model_id] = pipe_parent
        self._warmup_barrier.wait()
        logging.info("All Server Workers Initialized")

    def Inference(self, request, context):
        query_id = request.id
        model_id = request.model_id
        bs = request.bs
        seq_len = request.seq_len
        start_stamp = request.start_stamp
        # logging.info(time.time()-start_stamp)
        # start_stamp = time.time()
        qos_target = request.qos_target
        load_id = request.load_id
        query = Query(
            id=query_id,
            model_id=model_id,
            batch_size=bs,
            seq_len=seq_len,
            start_stamp=start_stamp,
            qos_target=qos_target,
            load_id=load_id,
        )
        query.set_op_pos(-1)
        self._pipes[model_id].send(
            (
                query.id,
                "new",
                0,
                query.start_pos,
                query.end_pos,
                query.batch_size,
                query.seq_len,
            )
        )
        self._barrier[0].wait()
        self._wr.writerow(
            np.array(
                [
                    query.id,
                    query.model_id,
                    query.batch_size,
                    query.seq_len,
                    query.load_id,
                    query.latency_ms(),
                ]
            )
        )
        self._result_file.flush()
        return service_pb2.Result(
            node_id=self._node_id, accepted=True, elapsed=query.latency_ms()
        )


class AbacusServer(service_pb2_grpc.DNNServerServicer):
    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._node_id = run_config.node_id
        self._warmup_barrier = mp.Barrier(run_config.total_models + 1)

        if self._run_config.policy == "Abacus" or self._run_config.policy == "mt-dnn" or self._run_config.policy == "mt-dnn2" or self._run_config.policy == "mt-dnn3" or self._run_config.policy == "linear" or self._run_config.policy == "tcp":
            self._barrier = []
            for i in range(run_config.total_models):
                self._barrier.append(mp.Barrier(i + 2))
        else:
            self._barrier = [mp.Barrier(2)]

        self._stop_event = mp.Event()
        self._workers = {}
        self._queues = {}
        self._pipes = {}
        self._qos_target = run_config.qos_target
        random.seed(0)
        self.start_up()

    def parsejson(self, model_name):
        file_path = os.path.join(
            "/home/onceas/wanna/Abacus/data/op_details", model_name+".json")
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            return content

    def culops(self, maps, start, end):
        res = [0, 0, 0, 0, 0, 0, 0, 0]
        # print(maps)
        for i in range(start-1, end):
            idx = 0
            for k in maps[i].keys():
                v = maps[i][k]
                res[idx] += v
                idx += 1
        return res

    # grpc Inference服务接口
    def Inference(self, request, context):
        query_id = request.id
        model_id = request.model_id
        bs = request.bs
        seq_len = request.seq_len
        # start_stamp = request.start_stamp
        start_stamp = time.time()
        # logging.info(time.time()-start_stamp)
        qos_target = request.qos_target
        load_id = request.load_id
        # logging.info(load_id)
        query = Query(
            id=query_id,
            model_id=model_id,
            batch_size=bs,
            seq_len=seq_len,
            start_stamp=start_stamp,
            qos_target=qos_target,
            load_id=load_id,
        )
        self._queues[model_id].put(query)
        # TODO Result？
        # logging.info("Abacus server Query received:{}".format(query))
        return service_pb2.Result(node_id=self._node_id, accepted=True)

    def send_query(self, id, model_id, batch_size, seq_len):
        self._queues[model_id].put(
            Query(
                id=id,
                model_id=model_id,
                batch_size=batch_size,
                seq_len=seq_len,
                qos_target=self._qos_target,
                load_id=0,
            )
        )

    # 启动Abacus server
    # 初始化所有DNN service的pipe
    # 初始化scheduler
    def start_up(self):
        logging.info("All Server Workers Initializing")
        self.op_details = {}
        for worker_id, model_id in enumerate(self._run_config.serve_combination):
            model_name = self._run_config.models_name[model_id]
            self.op_details[model_name] = self.parsejson(model_name)
            pipe_parent, pipe_child = mp.Pipe()
            model_worker = ServerWorker(
                self._run_config,
                model_name,
                self._run_config.supported_batchsize,
                self._run_config.supported_seqlen,
                pipe_child,
                self._barrier,
                self._warmup_barrier,
                worker_id,
            )
            model_worker.start()
            self._workers[model_id] = model_worker
            self._queues[model_id] = mp.Queue()
            self._pipes[model_id] = pipe_parent

        self._warmup_barrier.wait()
        logging.info("All Server Workers Initialized")

        logging.info("Scheduler Initializing")
        self._scheduler = Scheduler(
            run_config=self._run_config,
            barrier=self._barrier,
            queues=self._queues,
            pipes=self._pipes,
            stop_event=self._stop_event,
        )
        self._scheduler.start()
        logging.info("Scheduler Initialized")

    # poison分布的模拟请求生成
    def prepare_test_queries(self, total_queries=1000, average_duration=20):
        self._test_queries = []

        for i in range(total_queries):
            model_id = random.choice(self._run_config.serve_combination)

            sleep_duration = random.expovariate(average_duration)
            bs = random.choice(self._run_config.supported_batchsize)
            seq_len = (
                random.choice(
                    self._run_config.supported_seqlen) if model_id == 6 else 0
            )
            self._test_queries.append((model_id, sleep_duration, bs, seq_len))

    # 开始发送模拟请求
    def start_test(self):
        self.prepare_test_queries(
            self._run_config.total_queries, self._run_config.average_duration
        )
        i = 0
        for model_id, sleep_duration, bs, seq_len in self._test_queries:
            i += 1
            self.send_query(id=i, model_id=model_id,
                            batch_size=bs, seq_len=seq_len)
            time.sleep(sleep_duration)

    # 停止server
    # 当前queue都处理完了才会停止
    # 停止所有DNN service的pipe
    def stop_test(self):
        while not self.if_all_processed():
            logging.debug("Queries remain to be processed")
            time.sleep(0.01)
        self._stop_event.set()
        self._scheduler.join()

        for pipe in self._pipes.values():
            pipe.send((-1, "terminate", -1, -1, -1, -1, -1))
        for worker in self._workers.values():
            worker.join()

    # 判断请求队列是否都处理完了
    def if_all_processed(self):
        for queue in self._queues.values():
            if not queue.empty():
                return False
        return True


class Scheduler(Process):
    def __init__(
        self,
        run_config: RunConfig,
        barrier,
        queues,
        pipes,
        stop_event,
    ) -> None:
        super().__init__()
        self._policy = run_config.policy
        self._run_config = run_config
        # self._batch=run_config.batch
        self.op_details = {}
        self._threashold = run_config.threshold
        for worker_id, model_id in enumerate(run_config.serve_combination):
            model_name = run_config.models_name[model_id]
            self.op_details[model_name] = self.parsejson(model_name)
        if self._policy == "Abacus":
            self._search_ways = run_config.search_ways
            self._threashold = run_config.threshold

        if run_config.total_models == 2:
            if run_config.mig == 0:
                predictor_path = "model/2080Ti/2in7/all-"+self._run_config.predictor+".ckpt"
                log_path = "results/2080Ti/2in7/" + self._policy
                if run_config.platform == "cluster":
                    log_path = "results/cluster/" + self._policy
            elif run_config.mig == 2:
                predictor_path = "model/mig/2in4/all.ckpt"
                log_path = "results/mig/2in4/" + self._policy
            else:
                raise NotImplementedError
        elif run_config.total_models == 3:
            if run_config.mig == 0:
                predictor_path = "model/2080Ti/3in4/all.ckpt"
                log_path = "results/2080Ti/3in4/" + self._policy
            else:
                raise NotImplementedError
        elif run_config.total_models == 4:
            if run_config.mig == 0:
                if run_config.platform == "single":
                    predictor_path = "model/2080Ti/4in4/all.ckpt"
                    log_path = "results/2080Ti/4in4/" + self._policy
                elif run_config.platform == "cluster":
                    predictor_path = "model/cluster/4in4/all.ckpt"
                    log_path = "results/cluster/" + self._policy
            elif run_config.mig == 1:
                predictor_path = "model/mig/4in4/all.ckpt"
                log_path = "results/mig/4in4/" + self._policy
            else:
                raise NotImplementedError
        elif run_config.total_models == 1:
            if run_config.mig == 0:
                predictor_path = None
                log_path = "results/2080Ti/1in4/" + self._policy
            elif run_config.mig == 4:
                predictor_path = None
                log_path = "results/mig/1in4/" + self._policy
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        log_dir = os.path.join(run_config.path, log_path)
        os.makedirs(log_dir, exist_ok=True)

        self._serve_combination = run_config.serve_combination
        result_fname = ""
        for model_id in self._serve_combination:
            result_fname += run_config.models_name[model_id]
        result_fname += ".csv"
        self._result_path = os.path.join(log_dir, result_fname)
        # print(self._result_path)

        if predictor_path is not None:
            self._predictor_ckpt_path = os.path.join(
                run_config.path, predictor_path)
        else:
            self._predictor_ckpt_path = None
        self._abandon = run_config.abandon
        self._barrier = barrier
        self._queues = queues
        self._pipes = pipes
        self._stop_event = stop_event
        # TODO fix here
        if self._run_config.predictor == "layer":
            self._models_feature = len(self._run_config.models_len) + \
                4 * run_config.total_models
        elif self._run_config.predictor == "layer2":
            self._models_feature = len(self._run_config.models_len)-1 + \
                4 * run_config.total_models
        else:
            self._models_feature = 2 * 9
        # self._models_feature = len(Query.MODELS_LEN) * 8

    def run(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self._result_file = open(self._result_path, "w+")
        self._wr = csv.writer(self._result_file, dialect="excel")

        if self._predictor_ckpt_path is not None:
            logging.info(
                "Scheduler loading predictor from {}".format(
                    self._predictor_ckpt_path)
            )
            self._predictor = MLPregression(self._models_feature)
            self._predictor.load_state_dict(
                torch.load(self._predictor_ckpt_path, map_location="cpu")
            )
            self._predictor.eval()

            logging.info("Scheduler warmpup predictor")
            warmp_up_input = torch.zeros(16, self._models_feature)
            with torch.no_grad():
                for i in range(200):
                    warmp_up_output = self._predictor(warmp_up_input).numpy()
            logging.info("Scheduler end up warm up")

        logging.info("Scheduler Ready")
        self._scheduling_queries = {
            key: None for key in self._serve_combination}
        self._scheduled_queries = {
            key: None for key in self._serve_combination}

        result_header = ["query_id", "model_id", "bs", "seq_len", "latency"]

        if self._policy == "mt-dnn":
            self._schedule_func = self.mtdnn_schedule
            # 表示执行完的dnn数量(return)
            # the number of scheduled queries in last round of scheduling, equal with schedule_flag + 1
            self._schedule_flag = -1
            self._predicted_latency = 0
            # search times for the optimal operator group
            self._search_times = 0
            self._batch = False
            result_header = [
                "query_id",
                "model_id",
                "bs",
                "seq_len",
                "search_times",
                "load_id",
                "latency",
            ]
        elif self._policy == "mt-dnn2":
            self._schedule_func = self.mtdnn2_schedule
            # 表示执行完的dnn数量(return)
            # the number of scheduled queries in last round of scheduling, equal with schedule_flag + 1
            self._schedule_flag = -1
            self._predicted_latency = 0
            # search times for the optimal operator group
            self._search_times = 0
            result_header = [
                "query_id",
                "model_id",
                "bs",
                "seq_len",
                "search_times",
                "load_id",
                "latency",
            ]
        elif self._policy == "mt-dnn3":
            self._schedule_func = self.mtdnn3_schedule
            self._schedule_flag = -1
            self._predicted_latency = 0
            # search times for the optimal operator group
            self._search_times = 0
            # TODO fix here
            self._batch = True
            result_header = [
                "query_id",
                "model_id",
                "bs",
                "seq_len",
                "search_times",
                "load_id",
                "latency",
            ]
        elif self._policy == "linear":
            self._schedule_func = self.linear_schedule
            # 表示执行完的dnn数量(return)
            # the number of scheduled queries in last round of scheduling, equal with schedule_flag + 1
            self._schedule_flag = -1
            self._predicted_latency = 0
            # search times for the optimal operator group
            self._search_times = 0
            result_header = [
                "query_id",
                "model_id",
                "bs",
                "seq_len",
                "search_times",
                "load_id",
                "latency",
            ]
        elif self._policy == "tcp":
            self._schedule_func = self.tcp_schedule
            # 表示执行完的dnn数量(return)
            # the number of scheduled queries in last round of scheduling, equal with schedule_flag + 1
            self._schedule_flag = -1
            self._predicted_latency = 0
            # search times for the optimal operator group
            self._search_times = 0
            result_header = [
                "query_id",
                "model_id",
                "bs",
                "seq_len",
                "search_times",
                "load_id",
                "latency",
            ]
        elif self._policy == "Abacus":
            self._schedule_func = self.Abacus_schedule2
            # 表示执行完的dnn数量(return)
            # the number of scheduled queries in last round of scheduling, equal with schedule_flag + 1
            self._schedule_flag = -1
            self._predicted_latency = 0
            # search times for the optimal operator group
            self._search_times = 0
            result_header = [
                "query_id",
                "model_id",
                "bs",
                "seq_len",
                "search_times",
                "load_id",
                "latency",
            ]
        elif self._policy == "SJF":
            self._schedule_func = self.SJF_schedule
        elif self._policy == "FCFS":
            self._schedule_func = self.FCFS_schedule
        elif self._policy == "EDF":
            self._schedule_func = self.EDF_schedule
        else:
            raise NotImplementedError
        self._wr.writerow(result_header)
        self._result_file.flush()
        logging.info(
            "Performing scheudling with Policy: {}".format(self._policy))

        if self._policy == "FCFS" or self._policy == "SJF" or self._policy == "EDF":
            while not self._stop_event.is_set() or self.remain_schedule():
                self.pop_queries()
                self._schedule_func(self._abandon)
        elif self._policy == "Abacus":
            while not self._stop_event.is_set():
                self.pop_queries()
                self.Abacus_schedule2()
            self.Abacus_reset()
        elif self._policy == "mt-dnn":
            while not self._stop_event.is_set():
                self.pop_queries()
                self.mtdnn_schedule()
            self.Abacus_reset()
        elif self._policy == "mt-dnn2":
            while not self._stop_event.is_set():
                self.pop_queries()
                self.mtdnn2_schedule()
            self.Abacus_reset()
        elif self._policy == "mt-dnn3":
            while not self._stop_event.is_set():
                self.pop_queries()
                self.mtdnn3_schedule()
            self.Abacus_reset()
        elif self._policy == "linear":
            while not self._stop_event.is_set():
                self.pop_queries()
                self.linear_schedule()
            self.Abacus_reset()
        elif self._policy == "tcp":
            while not self._stop_event.is_set():
                self.pop_queries()
                self.tcp_schedule()
            self.Abacus_reset()

    def Abacus_schedule(self):
        waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        # logging.info("Abacus_schedule:{} {}".format(
        # waiting_scheduling, abandoned_scheduling))
        total_colocated = len(waiting_scheduling)

        if total_colocated > 0:
            # qos_id必须完整运行
            qos_id = waiting_scheduling[0][0]
            qos = waiting_scheduling[0][1] - self._predicted_latency
            qos_query: Query = self._scheduling_queries[qos_id]
            qos_query.set_op_pos(-1)

            # 只有一个，如果abandon，预测并判断丢弃，否则直接执行即可
            if total_colocated == 1:
                if self._abandon:
                    with torch.no_grad():
                        # 预测时间
                        inpuit_features = self.get_layer_feature(
                            0, 0, 1, qos_query=qos_query)[1]
                        self._predicted_latency = self._predictor(inpuit_features)[
                            0]
                    self.Abacus_reset()
                    # 剩余qos小于预测时间，丢弃
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    self.Abacus_reset()
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            0,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 0
            elif total_colocated >= 2:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                if total_colocated >= 3:
                    m_id = waiting_scheduling[2][0]
                    m_query: Query = self._scheduling_queries[m_id]
                else:
                    m_id = None
                    m_query = None
                if total_colocated >= 4:
                    r_id = waiting_scheduling[3][0]
                    r_query: Query = self._scheduling_queries[r_id]
                else:
                    r_id = None
                    r_query = None
                searched_query = None
                searched_pos = None

                # 第一次搜索，粒度为query
                features = self.get_query_feature(
                    total_colocated, qos_query, l_query, m_query, r_query
                )
                search_times = 1
                with torch.no_grad():
                    latencies = self._predictor(features).numpy()
                if total_colocated == 4:
                    if qos >= latencies[3]:
                        searched_query = 4
                    elif qos >= latencies[2]:
                        searched_query = 3
                    elif qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0
                elif total_colocated == 3:
                    if qos >= latencies[2]:
                        searched_query = 3
                    elif qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0
                elif total_colocated == 2:
                    if qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0

                # 第二次搜索，粒度为layer
                if searched_query == 4:
                    l_query.set_op_pos(-1)
                    m_query.set_op_pos(-1)
                    r_query.set_op_pos(-1)
                elif searched_query == 3:
                    if total_colocated >= 4:
                        # search in r query
                        l_query.set_op_pos(-1)
                        m_query.set_op_pos(-1)
                        start_pos = r_query.end_pos
                        end_pos = r_query.op_len
                        while start_pos < end_pos:
                            search_times += 1
                            search_ways = (
                                self._search_ways
                                if (end_pos - start_pos) > self._search_ways
                                else (end_pos - start_pos)
                            )
                            op_poses, features = self.get_layer_feature(
                                start_pos=start_pos,
                                end_pos=end_pos,
                                search_ways=search_ways,
                                qos_query=qos_query,
                                l_query=l_query,
                                m_query=m_query,
                                r_query=r_query,
                            )
                            with torch.no_grad():
                                latencies = self._predictor(features).numpy()
                            # 搜索到哪个end operator结束
                            for i in range(search_ways):
                                headroom = qos - latencies[i]
                                if headroom < 0:
                                    end_pos = op_poses[i]
                                    if i == 0:
                                        searched_pos = op_poses[i]
                                        break
                                else:
                                    if headroom < self._threashold:
                                        searched_pos = op_poses[i]
                                        self._predicted_latency = latencies[i]
                                        break
                                    else:
                                        start_pos = op_poses[i] + 1
                                if i == search_ways - 1:
                                    searched_pos = op_poses[i]
                            if searched_pos is not None:
                                break
                    else:
                        # only l and m query
                        l_query.set_op_pos(-1)
                        m_query.set_op_pos(-1)
                elif searched_query == 2:
                    l_query.set_op_pos(-1)
                    if total_colocated >= 3:
                        # search in m query
                        start_pos = m_query.end_pos
                        end_pos = m_query.op_len
                        while start_pos < end_pos:
                            search_times += 1
                            search_ways = (
                                self._search_ways
                                if (end_pos - start_pos) > self._search_ways
                                else (end_pos - start_pos)
                            )
                            op_poses, features = self.get_layer_feature(
                                start_pos=start_pos,
                                end_pos=end_pos,
                                search_ways=search_ways,
                                qos_query=qos_query,
                                l_query=l_query,
                                m_query=m_query,
                            )
                            with torch.no_grad():
                                latencies = self._predictor(features).numpy()
                            for i in range(search_ways):
                                headroom = qos - latencies[i]
                                if headroom < 0:
                                    end_pos = op_poses[i]
                                    if i == 0:
                                        searched_pos = op_poses[i]
                                        break
                                else:
                                    if headroom < self._threashold:
                                        searched_pos = op_poses[i]
                                        self._predicted_latency = latencies[i]
                                        break
                                    else:
                                        start_pos = op_poses[i] + 1
                                if i == search_ways - 1:
                                    searched_pos = op_poses[i]
                            if searched_pos is not None:
                                break
                elif searched_query == 1:
                    # search in l query
                    start_pos = l_query.end_pos
                    end_pos = l_query.op_len

                    while start_pos < end_pos:
                        search_times += 1
                        search_ways = (
                            self._search_ways
                            if (end_pos - start_pos) > self._search_ways
                            else (end_pos - start_pos)
                        )
                        op_poses, features = self.get_layer_feature(
                            start_pos=start_pos,
                            end_pos=end_pos,
                            search_ways=search_ways,
                            qos_query=qos_query,
                            l_query=l_query,
                        )
                        with torch.no_grad():
                            latencies = self._predictor(features).numpy()
                        for i in range(search_ways):
                            headroom = qos - latencies[i]
                            if headroom < 0:
                                end_pos = op_poses[i]
                                if i == 0:
                                    searched_pos = op_poses[i]
                            else:
                                if headroom < self._threashold:
                                    searched_pos = op_poses[i]
                                    self._predicted_latency = latencies[i]
                                    break
                                else:
                                    start_pos = op_poses[i] + 1
                            if i == search_ways - 1:
                                searched_pos = op_poses[-1]
                        if searched_pos is not None:
                            break

                self.Abacus_reset(search_times=search_times)

                if searched_query == 0:
                    if self._abandon:
                        abandoned_scheduling.append(qos_id)
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                elif searched_query == 1:
                    l_query.set_op_pos(searched_pos)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    if l_query.if_processed():
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self.clean_scheduling_queries(model_ids=[qos_id, l_id])
                    else:
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 1
                elif searched_query == 2:
                    if total_colocated >= 3:
                        m_query.set_op_pos(searched_pos)
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                2,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._pipes[l_id].send(
                            (
                                l_query.id,
                                l_query.state,
                                2,
                                l_query.start_pos,
                                l_query.end_pos,
                                l_query.batch_size,
                                l_query.seq_len,
                            )
                        )
                        self._pipes[m_id].send(
                            (
                                m_query.id,
                                m_query.state,
                                2,
                                m_query.start_pos,
                                m_query.end_pos,
                                m_query.batch_size,
                                m_query.seq_len,
                            )
                        )
                        if m_query.if_processed():
                            self._scheduled_queries[qos_id] = qos_query
                            self._scheduled_queries[l_id] = l_query
                            self._scheduled_queries[m_id] = m_query
                            self.clean_scheduling_queries(
                                model_ids=[qos_id, l_id, m_id])
                        else:
                            self._scheduled_queries[qos_id] = qos_query
                            self._scheduled_queries[l_id] = l_query
                            self.clean_scheduling_queries(
                                model_ids=[qos_id, l_id])
                        self._schedule_flag = 2
                    else:  # 2
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                1,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._pipes[l_id].send(
                            (
                                l_query.id,
                                l_query.state,
                                1,
                                l_query.start_pos,
                                l_query.end_pos,
                                l_query.batch_size,
                                l_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self.clean_scheduling_queries(model_ids=[qos_id, l_id])
                        self._schedule_flag = 1

                elif searched_query == 3:
                    if total_colocated >= 4:
                        r_query.set_op_pos(searched_pos)
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                3,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._pipes[l_id].send(
                            (
                                l_query.id,
                                l_query.state,
                                3,
                                l_query.start_pos,
                                l_query.end_pos,
                                l_query.batch_size,
                                l_query.seq_len,
                            )
                        )
                        self._pipes[m_id].send(
                            (
                                m_query.id,
                                m_query.state,
                                3,
                                m_query.start_pos,
                                m_query.end_pos,
                                m_query.batch_size,
                                m_query.seq_len,
                            )
                        )
                        self._pipes[r_id].send(
                            (
                                r_query.id,
                                r_query.state,
                                3,
                                r_query.start_pos,
                                r_query.end_pos,
                                r_query.batch_size,
                                r_query.seq_len,
                            )
                        )
                        if r_query.if_processed():
                            self._scheduled_queries[qos_id] = qos_query
                            self._scheduled_queries[l_id] = l_query
                            self._scheduled_queries[m_id] = m_query
                            self._scheduled_queries[r_id] = r_query
                            self.clean_scheduling_queries(
                                model_ids=[qos_id, l_id, m_id, r_id]
                            )
                        else:
                            self._scheduled_queries[qos_id] = qos_query
                            self._scheduled_queries[l_id] = l_query
                            self._scheduled_queries[m_id] = m_query
                            self.clean_scheduling_queries(
                                model_ids=[qos_id, l_id, m_id])
                        self._schedule_flag = 3
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                2,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._pipes[l_id].send(
                            (
                                l_query.id,
                                l_query.state,
                                2,
                                l_query.start_pos,
                                l_query.end_pos,
                                l_query.batch_size,
                                l_query.seq_len,
                            )
                        )
                        self._pipes[m_id].send(
                            (
                                m_query.id,
                                m_query.state,
                                2,
                                m_query.start_pos,
                                m_query.end_pos,
                                m_query.batch_size,
                                m_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self._scheduled_queries[m_id] = m_query
                        self.clean_scheduling_queries(
                            model_ids=[qos_id, l_id, m_id])
                        self._schedule_flag = 2
                elif searched_query == 4:
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            3,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            3,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    self._pipes[m_id].send(
                        (
                            m_query.id,
                            m_query.state,
                            3,
                            m_query.start_pos,
                            m_query.end_pos,
                            m_query.batch_size,
                            m_query.seq_len,
                        )
                    )
                    self._pipes[r_id].send(
                        (
                            r_query.id,
                            r_query.state,
                            3,
                            r_query.start_pos,
                            r_query.end_pos,
                            r_query.batch_size,
                            r_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self._scheduled_queries[l_id] = l_query
                    self._scheduled_queries[m_id] = m_query
                    self._scheduled_queries[r_id] = r_query
                    self.clean_scheduling_queries(
                        model_ids=[qos_id, l_id, m_id, r_id])
                    self._schedule_flag = 3
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        0,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    def Abacus_schedule2(self):
        waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        # logging.info("Abacus_schedule:{} {}".format(
        # waiting_scheduling, abandoned_scheduling))
        total_colocated = len(waiting_scheduling)

        if total_colocated > 0:
            # qos_id必须完整运行
            qos_id = waiting_scheduling[0][0]
            qos = waiting_scheduling[0][1] - self._predicted_latency
            qos_query: Query = self._scheduling_queries[qos_id]
            qos_query.set_op_pos(-1)

            # 只有一个，如果abandon，预测并判断丢弃，否则直接执行即可
            if total_colocated == 1:
                if self._abandon:
                    with torch.no_grad():
                        # 预测时间
                        inpuit_features = self.get_layer_feature(
                            0, 0, 1, qos_query=qos_query)[1]
                        self._predicted_latency = self._predictor(inpuit_features)[
                            0]
                    self.Abacus_reset()
                    # 剩余qos小于预测时间，丢弃
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    self.Abacus_reset()
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            0,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 0
            elif total_colocated >= 2:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                m_query = None
                r_query = None
                searched_query = None
                searched_pos = None

                # 第一次搜索，粒度为query
                features = self.get_query_feature(
                    total_colocated, qos_query, l_query, m_query, r_query
                )
                search_times = 1
                with torch.no_grad():
                    latencies = self._predictor(features).numpy()
                if total_colocated == 2:
                    if qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0

                # 第二次搜索，粒度为layer
                if searched_query == 1:
                    # search in l query
                    start_pos = l_query.end_pos
                    end_pos = l_query.op_len

                    while start_pos < end_pos:
                        search_times += 1
                        search_ways = (
                            self._search_ways
                            if (end_pos - start_pos) > self._search_ways
                            else (end_pos - start_pos)
                        )
                        op_poses, features = self.get_layer_feature(
                            start_pos=start_pos,
                            end_pos=end_pos,
                            search_ways=search_ways,
                            qos_query=qos_query,
                            l_query=l_query,
                        )
                        with torch.no_grad():
                            latencies = self._predictor(features).numpy()
                        for i in range(search_ways):
                            headroom = qos - latencies[i]
                            if headroom < 0:
                                end_pos = op_poses[i]
                                if i == 0:
                                    searched_pos = op_poses[i]
                            else:
                                if headroom < self._threashold:
                                    searched_pos = op_poses[i]
                                    self._predicted_latency = latencies[i]
                                    break
                                else:
                                    start_pos = op_poses[i] + 1
                            if i == search_ways - 1:
                                searched_pos = op_poses[-1]
                        if searched_pos is not None:
                            break

                self.Abacus_reset(search_times=search_times)

                if searched_query == 0:
                    if self._abandon:
                        abandoned_scheduling.append(qos_id)
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                elif searched_query == 1:
                    l_query.set_op_pos(searched_pos)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    if l_query.if_processed():
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self.clean_scheduling_queries(
                            model_ids=[qos_id, l_id])
                    else:
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 1
                elif searched_query == 2:
                    l_query.set_op_pos(-1)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self._scheduled_queries[l_id] = l_query
                    self.clean_scheduling_queries(model_ids=[qos_id, l_id])
                    self._schedule_flag = 1

                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        0,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    def tcp_schedule(self):
        waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        # logging.info("Abacus_schedule:{} {}".format(
        # waiting_scheduling, abandoned_scheduling))
        total_colocated = len(waiting_scheduling)

        # logging.info("start tcp schedule")
        if total_colocated > 0:
            # qos_id必须完整运行
            qos_id = waiting_scheduling[0][0]
            qos = waiting_scheduling[0][1] - self._predicted_latency
            qos_query: Query = self._scheduling_queries[qos_id]
            qos_query.set_op_pos(-1)

            # 只有一个，如果abandon，预测并判断丢弃，否则直接执行即可
            if total_colocated == 1:
                if self._abandon:
                    with torch.no_grad():
                        # 预测时间
                        inpuit_features = self.get_layer_feature(
                            0, 0, 1, qos_query=qos_query)[1]
                        self._predicted_latency = self._predictor(inpuit_features)[
                            0]
                    self.Abacus_reset()
                    # 剩余qos小于预测时间，丢弃
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    self.Abacus_reset()
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            0,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 0
            elif total_colocated >= 2:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                m_query = None
                r_query = None
                searched_query = None
                searched_pos = None

                # 第一次搜索，粒度为query
                features = self.get_query_feature(
                    total_colocated, qos_query, l_query, m_query, r_query
                )
                search_times = 1
                with torch.no_grad():
                    latencies = self._predictor(features).numpy()
                if total_colocated == 2:
                    if qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0

                # 第二次搜索，粒度为layer
                if searched_query == 1:
                    # search in l query
                    search_ways = 1
                    # logging.info("start get layer tcp ")
                    searched_pos = self.get_layer_tcp_phase(
                        search_ways=search_ways,
                        qos_query=qos_query,
                        l_query=l_query,
                    )
                    # logging.info("searched_pos:{}".format(searched_pos))

                self.Abacus_reset(search_times=search_times)

                if searched_query == 0:
                    if self._abandon:
                        abandoned_scheduling.append(qos_id)
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                elif searched_query == 1:
                    l_query.set_op_pos(searched_pos)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    if l_query.if_processed():
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self.clean_scheduling_queries(
                            model_ids=[qos_id, l_id])
                    else:
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 1
                elif searched_query == 2:
                    l_query.set_op_pos(-1)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self._scheduled_queries[l_id] = l_query
                    self.clean_scheduling_queries(model_ids=[qos_id, l_id])
                    self._schedule_flag = 1

                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        0,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    # search_ways表示有多少种可能
    # 1-search_ways每个分别往前推进step，即start-end，end是在变化的
    def get_layer_tcp_phase(
        self,
        search_ways,
        qos_query: Query,
        l_query: Query = None,
    ):

        lo = 0
        if l_query.end_pos != 0:
            lo = l_query.end_pos
        hi = l_query.op_len
        lens = len(self._run_config.models_id)
        idx = 1
        now = lo
        poses = []
        while True:
            now = now+idx
            poses.append(now)
            idx = 2*idx
            if now+idx >= hi:
                poses.append(min(now+idx, hi))
                break

        # self._models_feature = len(Query.MODELS_LEN) + 4 * run_config.total_models
        # input_feature size为：search_ways × self._models_feature
        input_feature = torch.zeros(1, self._models_feature)

        # 相应bitmap设置为1
        input_feature[0][qos_query.model_id] += 1
        # model features
        input_feature[0][lens] = qos_query.start_pos
        input_feature[0][lens+1] = qos_query.end_pos
        input_feature[0][lens+2] = qos_query.batch_size
        input_feature[0][lens+3] = qos_query.seq_len

        latency = None
        start = None
        if l_query is not None:
            input_feature[0][l_query.model_id] += 1
            input_feature[0][lens+4] = l_query.start_pos
            # input_feature[0:search_ways, lens+5] = l_query.end_pos
            input_feature[0][lens+6] = l_query.batch_size
            input_feature[0][lens+7] = l_query.seq_len

            for i in range(len(poses)):
                start_pos = poses[i]
                input_feature[0][lens+5] = start_pos
                with torch.no_grad():
                    latency = self._predictor(input_feature).numpy()[0]
                    self._predicted_latency = latency
                qos = qos_query.get_headromm()
                if qos < 0:
                    return poses[i]
                if latency > qos or latency > qos-self._threashold:
                    break
                else:
                    start = poses[i]
            if start is None:
                start = poses[0]
            while start < hi:
                start += 1
                input_feature[0][lens+5] = start
                with torch.no_grad():
                    latency = self._predictor(input_feature).numpy()[0]
                    self._predicted_latency = latency
                qos = qos_query.get_headromm()
                if qos < 0:
                    return start
                if latency > qos:
                    break
        logging.debug("input feature: {}".format(input_feature))
        return start

    def linear_schedule2(self):
        waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        # logging.info("Abacus_schedule:{} {}".format(
        # waiting_scheduling, abandoned_scheduling))
        total_colocated = len(waiting_scheduling)

        # logging.info("start mtdnn_schedule")
        if total_colocated > 0:
            # qos_id必须完整运行
            qos_id = waiting_scheduling[0][0]
            # deadline余量
            qos = waiting_scheduling[0][1] - self._predicted_latency
            qos_query: Query = self._scheduling_queries[qos_id]
            qos_query.set_op_pos(-1)
            # 只有一个，如果abandon，预测并判断丢弃，否则直接执行即可
            if total_colocated == 1:
                if self._abandon:
                    with torch.no_grad():
                        # 预测时间
                        inpuit_features = self.get_layer_feature(
                            0, 0, 1, qos_query=qos_query)[1]
                        self._predicted_latency = self._predictor(inpuit_features)[
                            0]
                    self.Abacus_reset()
                    # 剩余qos小于预测时间，丢弃
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    self.Abacus_reset()
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            0,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 0
            elif total_colocated >= 2:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                end_pos = self.linear_search(qos_query, l_query)
                # logging.info("end_pos:{}".format(end_pos))
                self.Abacus_reset()
                barrir_tag = 1
                if end_pos == 0:
                    barrir_tag = 0
                self._pipes[qos_id].send(
                    (
                        qos_query.id,
                        qos_query.state,
                        barrir_tag,
                        qos_query.start_pos,
                        qos_query.end_pos,
                        qos_query.batch_size,
                        qos_query.seq_len,
                    )
                )
                if end_pos != 0:
                    l_query.set_op_pos(end_pos)
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                if l_query.if_processed():
                    self._scheduled_queries[qos_id] = qos_query
                    self._scheduled_queries[l_id] = l_query
                    self.clean_scheduling_queries(model_ids=[qos_id, l_id])
                else:
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                self._schedule_flag = barrir_tag
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        0,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    def linear_schedule(self):
        waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        # logging.info("Abacus_schedule:{} {}".format(
        # waiting_scheduling, abandoned_scheduling))
        total_colocated = len(waiting_scheduling)

        # logging.info("start tcp schedule")
        if total_colocated > 0:
            # qos_id必须完整运行
            qos_id = waiting_scheduling[0][0]
            qos = waiting_scheduling[0][1] - self._predicted_latency
            qos_query: Query = self._scheduling_queries[qos_id]
            qos_query.set_op_pos(-1)

            # 只有一个，如果abandon，预测并判断丢弃，否则直接执行即可
            if total_colocated == 1:
                if self._abandon:
                    with torch.no_grad():
                        # 预测时间
                        inpuit_features = self.get_layer_feature(
                            0, 0, 1, qos_query=qos_query)[1]
                        self._predicted_latency = self._predictor(inpuit_features)[
                            0]
                    self.Abacus_reset()
                    # 剩余qos小于预测时间，丢弃
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    self.Abacus_reset()
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            0,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 0
            elif total_colocated >= 2:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                m_query = None
                r_query = None
                searched_query = None
                searched_pos = None

                # 第一次搜索，粒度为query
                features = self.get_query_feature(
                    total_colocated, qos_query, l_query, m_query, r_query
                )
                search_times = 1
                with torch.no_grad():
                    latencies = self._predictor(features).numpy()
                if total_colocated == 2:
                    if qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0

                # 第二次搜索，粒度为layer
                if searched_query == 1:
                    # search in l query
                    search_ways = 1
                    searched_pos = self.linear_search(
                        qos_query=qos_query,
                        l_query=l_query,
                    )
                    # logging.info("searched_pos:{}".format(searched_pos))

                self.Abacus_reset(search_times=search_times)

                if searched_query == 0:
                    if self._abandon:
                        abandoned_scheduling.append(qos_id)
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                elif searched_query == 1:
                    l_query.set_op_pos(searched_pos)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    if l_query.if_processed():
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self.clean_scheduling_queries(
                            model_ids=[qos_id, l_id])
                    else:
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 1
                elif searched_query == 2:
                    l_query.set_op_pos(-1)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self._scheduled_queries[l_id] = l_query
                    self.clean_scheduling_queries(model_ids=[qos_id, l_id])
                    self._schedule_flag = 1

                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        0,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    def linear_search(self, qos_query: Query, l_query: Query):
        # logging.info("qos {} {},l_query {} {}".format(
        # qos_query.start_pos, qos_query.end_pos, l_query.start_pos, l_query.end_pos))
        lo = l_query.end_pos
        hi = l_query.op_len
        latency = None
        target_pos = lo
        for i in range(lo, hi+1):
            qos = qos_query.get_headromm()
            if qos < 0:
                return lo
            with torch.no_grad():
                latency = self._predictor(self.build_feature(
                    qos_query, l_query, end_pos=i)).numpy()[0]
            self._predicted_latency = latency
            # logging.info("{} {} {} {} {}".format(qos, latency, lo, mid, hi))
            if qos < latency:
                target_pos = max(lo, i-1)
                break
            elif qos-latency < self._threashold:
                target_pos = i
                break
        with torch.no_grad():
            self._predicted_latency = self._predictor(self.build_feature(
                qos_query, l_query, end_pos=target_pos)).numpy()[0]
        return target_pos

    def mtdnn_schedule2(self):
        waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        # logging.info("Abacus_schedule:{} {}".format(
        # waiting_scheduling, abandoned_scheduling))
        total_colocated = len(waiting_scheduling)

        # logging.info("start mtdnn_schedule")
        if total_colocated > 0:
            # qos_id必须完整运行
            qos_id = waiting_scheduling[0][0]
            # deadline余量
            qos = waiting_scheduling[0][1] - self._predicted_latency
            qos_query: Query = self._scheduling_queries[qos_id]
            qos_query.set_op_pos(-1)
            # 只有一个，如果abandon，预测并判断丢弃，否则直接执行即可
            if total_colocated == 1:
                if self._abandon:
                    with torch.no_grad():
                        # 预测时间
                        inpuit_features = self.get_layer_feature(
                            0, 0, 1, qos_query=qos_query)[1]
                        self._predicted_latency = self._predictor(inpuit_features)[
                            0]
                    self.Abacus_reset()
                    # 剩余qos小于预测时间，丢弃
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    self.Abacus_reset()
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            0,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 0
            elif total_colocated >= 2:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                end_pos = self.binary_search(qos_query, l_query)
                # end_pos = self.get_layer_tcp_phase(2,qos_query, l_query)
                # logging.info("end_pos:{}".format(end_pos))
                self.Abacus_reset()
                barrir_tag = 1
                if end_pos == 0:
                    barrir_tag = 0
                self._pipes[qos_id].send(
                    (
                        qos_query.id,
                        qos_query.state,
                        barrir_tag,
                        qos_query.start_pos,
                        qos_query.end_pos,
                        qos_query.batch_size,
                        qos_query.seq_len,
                    )
                )
                if end_pos != 0:
                    l_query.set_op_pos(end_pos)
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                if l_query.if_processed():
                    self._scheduled_queries[qos_id] = qos_query
                    self._scheduled_queries[l_id] = l_query
                    self.clean_scheduling_queries(model_ids=[qos_id, l_id])
                else:
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                self._schedule_flag = barrir_tag
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        0,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    def mtdnn_schedule(self):
        waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        # logging.info("mtdnn_schedule:{} {}".format(
        #     waiting_scheduling, abandoned_scheduling))
        # for i in range(len(waiting_scheduling)):
        #     logging.info("query:{} {}".format(
        #         self._scheduling_queries[waiting_scheduling[i][0]], waiting_scheduling[i][1]))
        total_colocated = len(waiting_scheduling)

        # logging.info("start tcp schedule")
        if total_colocated > 0:
            # qos_id必须完整运行
            qos_id = waiting_scheduling[0][0]
            qos = waiting_scheduling[0][1] - self._predicted_latency
            qos_query: Query = self._scheduling_queries[qos_id]
            qos_query.set_op_pos(-1)

            # 只有一个，如果abandon，预测并判断丢弃，否则直接执行即可
            if total_colocated == 1:
                if self._abandon:
                    with torch.no_grad():
                        # 预测时间
                        inpuit_features = self.get_layer_feature(
                            0, 0, 1, qos_query=qos_query)[1]
                        self._predicted_latency = self._predictor(inpuit_features)[
                            0]
                    self.Abacus_reset()
                    # 剩余qos小于预测时间，丢弃
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    self.Abacus_reset()
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            0,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 0
            elif total_colocated >= 2:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                m_query = None
                r_query = None
                searched_query = None
                searched_pos = None

                # 第一次搜索，粒度为query
                features = self.get_query_feature(
                    total_colocated, qos_query, l_query, m_query, r_query
                )
                search_times = 1
                with torch.no_grad():
                    latencies = self._predictor(features).numpy()
                if total_colocated == 2:
                    if qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0

                # 第二次搜索，粒度为layer
                if searched_query == 1:
                    # search in l query
                    search_ways = 1
                    searched_pos = self.binary_search(
                        qos_query=qos_query,
                        l_query=l_query,
                    )
                    # logging.info(
                    #     "searched_query=1 {}  {}  searched_pos:{}".format(qos_query, l_query, searched_pos))
                self.Abacus_reset(search_times=search_times)
                if searched_query == 0:
                    if self._abandon:
                        abandoned_scheduling.append(qos_id)
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                elif searched_query == 1:
                    l_query.set_op_pos(searched_pos)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    if l_query.if_processed():
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self.clean_scheduling_queries(
                            model_ids=[qos_id, l_id])
                    else:
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 1
                elif searched_query == 2:
                    l_query.set_op_pos(-1)
                    # logging.info(
                    #     "searched_query=2  qos_query:{}  l_query:{}".format(qos_query, l_query))
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self._scheduled_queries[l_id] = l_query
                    self.clean_scheduling_queries(model_ids=[qos_id, l_id])
                    self._schedule_flag = 1

                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        0,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    def mtdnn3_schedule(self):
        waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        # logging.info("Abacus_schedule:{} {}".format(
        # waiting_scheduling, abandoned_scheduling))
        total_colocated = len(waiting_scheduling)

        # logging.info("start tcp schedule")
        if total_colocated > 0:
            # qos_id必须完整运行
            qos_id = waiting_scheduling[0][0]
            qos = waiting_scheduling[0][1] - self._predicted_latency
            qos_query: Query = self._scheduling_queries[qos_id]
            # qos_query.set_op_pos(-1)

            # 只有一个，如果abandon，预测并判断丢弃，否则直接执行即可
            if total_colocated == 1:
                qos_query.set_op_pos(-1)
                if self._abandon:
                    with torch.no_grad():
                        # 预测时间
                        inpuit_features = self.get_layer_feature(
                            0, 0, 1, qos_query=qos_query)[1]
                        self._predicted_latency = self._predictor(inpuit_features)[
                            0]
                    self.Abacus_reset()
                    # 剩余qos小于预测时间，丢弃
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    self.Abacus_reset()
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            0,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 0
            elif total_colocated >= 2:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                m_query = None
                r_query = None
                searched_query = None
                searched_pos = None

                # logging.info("recv qos:{}".format(qos_query))
                # logging.info("recv l:{}".format(l_query))

                # TODO add batch process
                if self._batch:
                    # 当两个 新来的 大batch组合时（测试不同gpu上不同组合的转折点），使用串行执行代替并行执行
                    if qos_query.batch_size >= 16 and qos_query.end_pos == 0 and l_query.batch_size >= 16:
                        qos_query.set_op_pos(-1)
                        l_query.set_op_pos(-1)

                        # qos_query single execute
                        with torch.no_grad():
                            # 预测时间
                            inpuit_features = self.get_layer_feature(
                                0, 0, 1, qos_query=qos_query)[1]
                            self._predicted_latency = self._predictor(inpuit_features)[
                                0]
                        if self._abandon and qos < self._predicted_latency:
                            self.Abacus_reset()
                            # 剩余qos小于预测时间，丢弃
                            abandoned_scheduling.append(qos_id)
                            self._predicted_latency = 0
                        else:
                            self.Abacus_reset()
                            self._pipes[qos_id].send(
                                (
                                    qos_query.id,
                                    qos_query.state,
                                    0,
                                    qos_query.start_pos,
                                    qos_query.end_pos,
                                    qos_query.batch_size,
                                    qos_query.seq_len,
                                )
                            )
                            self._scheduled_queries[qos_id] = qos_query
                            self.clean_scheduling_queries(model_ids=qos_id)
                            self._schedule_flag = 0
                            self._result_file.flush()
                        l_qos = waiting_scheduling[1][1]
                        with torch.no_grad():
                            # 预测时间
                            inpuit_features = self.get_layer_feature(
                                0, 0, 1, qos_query=l_query)[1]
                            self._predicted_latency = self._predictor(inpuit_features)[
                                0]
                        if self._abandon and l_qos < self._predicted_latency:
                            self.Abacus_reset()
                            abandoned_scheduling.append(l_id)
                            self._predicted_latency = 0
                        else:
                            self.Abacus_reset(search_times=0)
                            # logging.info("l_query1:{}".format(l_query))
                            # logging.info("l_query2:{}".format(l_query))
                            self._pipes[l_id].send(
                                (
                                    l_query.id,
                                    l_query.state,
                                    0,
                                    l_query.start_pos,
                                    l_query.end_pos,
                                    l_query.batch_size,
                                    l_query.seq_len,
                                ))
                            self._scheduled_queries[l_id] = l_query
                            self.clean_scheduling_queries(model_ids=l_id)
                            self._schedule_flag = 0
                        for model_id in abandoned_scheduling:
                            query = self._scheduling_queries[model_id]
                            self._wr.writerow(
                                np.array(
                                    [
                                        query.id,
                                        query.model_id,
                                        query.batch_size,
                                        query.seq_len,
                                        0,
                                        query.load_id,
                                        -1,
                                    ]
                                )
                            )
                            self.clean_scheduling_queries(model_ids=model_id)
                        self._result_file.flush()
                        return

                qos_query.set_op_pos(-1)
                # 第一次搜索，粒度为query
                features = self.get_query_feature(
                    total_colocated, qos_query, l_query, m_query, r_query
                )
                search_times = 1
                with torch.no_grad():
                    latencies = self._predictor(features).numpy()
                if total_colocated == 2:
                    if qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0

                # 第二次搜索，粒度为layer
                if searched_query == 1:
                    # search in l query
                    search_ways = 1
                    # logging.info("start get layer tcp ")
                    searched_pos = self.binary_search(
                        qos_query=qos_query,
                        l_query=l_query,
                    )
                    # logging.info("searched_pos:{}".format(searched_pos))

                self.Abacus_reset(search_times=search_times)

                if searched_query == 0:
                    if self._abandon:
                        abandoned_scheduling.append(qos_id)
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                elif searched_query == 1:
                    l_query.set_op_pos(searched_pos)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    if l_query.if_processed():
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self.clean_scheduling_queries(
                            model_ids=[qos_id, l_id])
                    else:
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 1
                elif searched_query == 2:
                    l_query.set_op_pos(-1)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self._scheduled_queries[l_id] = l_query
                    self.clean_scheduling_queries(model_ids=[qos_id, l_id])
                    self._schedule_flag = 1

                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        0,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    def mtdnn2_schedule(self):
        # waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        waiting_scheduling = {}
        abandoned_scheduling = []
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if self._abandon == True:
                if headroom > 0:
                    query.set_op_pos(-1)
                    arrival_stamp = query.start_stamp
                    waiting_scheduling[model_id] = arrival_stamp
                else:
                    abandoned_scheduling.append(model_id)
            else:
                query.set_op_pos(-1)
                arrival_stamp = query.start_stamp
                waiting_scheduling[model_id] = arrival_stamp

        # 按照到达时间进行排序
        waiting_scheduling = sorted(
            waiting_scheduling.items(), key=lambda x: x[1])
        total_colocated = len(waiting_scheduling)

        # logging.info("start tcp schedule")
        if total_colocated > 0:
            # qos_id必须完整运行
            qos_id = waiting_scheduling[0][0]
            qos = waiting_scheduling[0][1] - self._predicted_latency
            qos_query: Query = self._scheduling_queries[qos_id]
            qos_query.set_op_pos(-1)

            # 只有一个，如果abandon，预测并判断丢弃，否则直接执行即可
            if total_colocated == 1:
                if self._abandon:
                    with torch.no_grad():
                        # 预测时间
                        inpuit_features = self.get_layer_feature(
                            0, 0, 1, qos_query=qos_query)[1]
                        self._predicted_latency = self._predictor(inpuit_features)[
                            0]
                    self.Abacus_reset()
                    # 剩余qos小于预测时间，丢弃
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    self.Abacus_reset()
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            0,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 0
            elif total_colocated >= 2:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                m_query = None
                r_query = None
                searched_query = None
                searched_pos = None

                # 第一次搜索，粒度为query
                features = self.get_query_feature(
                    total_colocated, qos_query, l_query, m_query, r_query
                )
                search_times = 1
                with torch.no_grad():
                    latencies = self._predictor(features).numpy()
                if total_colocated == 2:
                    if qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0

                # 第二次搜索，粒度为layer
                if searched_query == 1:
                    # search in l query
                    search_ways = 1
                    # logging.info("start get layer tcp ")
                    searched_pos = self.binary_search(
                        qos_query=qos_query,
                        l_query=l_query,
                    )
                    # logging.info("searched_pos:{}".format(searched_pos))

                self.Abacus_reset(search_times=search_times)

                if searched_query == 0:
                    if self._abandon:
                        abandoned_scheduling.append(qos_id)
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                elif searched_query == 1:
                    l_query.set_op_pos(searched_pos)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    if l_query.if_processed():
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self.clean_scheduling_queries(
                            model_ids=[qos_id, l_id])
                    else:
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduling_queries(model_ids=qos_id)
                    self._schedule_flag = 1
                elif searched_query == 2:
                    l_query.set_op_pos(-1)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self._scheduled_queries[l_id] = l_query
                    self.clean_scheduling_queries(model_ids=[qos_id, l_id])
                    self._schedule_flag = 1

                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        0,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    def build_feature(self, qos_query: Query, l_query: Query, end_pos):
        input_feature = torch.zeros(1, self._models_feature)
        input_feature[0][qos_query.model_id] = 1
        input_feature[0][l_query.model_id] = 1

        lens = len(self._run_config.models_id)
        input_feature[0][lens] = qos_query.start_pos
        input_feature[0][lens+1] = qos_query.end_pos
        input_feature[0][lens+2] = qos_query.batch_size
        input_feature[0][lens+3] = qos_query.seq_len

        input_feature[0][lens+4] = l_query.start_pos
        input_feature[0][lens+5] = end_pos
        input_feature[0][lens+6] = l_query.batch_size
        input_feature[0][lens+7] = l_query.seq_len
        return input_feature

    def binary_search(self, qos_query: Query, l_query: Query):
        # logging.info("qos {} {},l_query {} {}".format(
        # qos_query.start_pos, qos_query.end_pos, l_query.start_pos, l_query.end_pos))
        lo = l_query.end_pos
        hi = l_query.op_len
        latency = None
        mid = None

        input_data = self.build_feature(qos_query, l_query, end_pos=0)
        lens = len(self._run_config.models_id)
        times = 0
        while lo < hi and times < 3:
            times += 1
            qos = qos_query.get_headromm()
            # if qos < 0:
            #     return 0
            mid = (lo+hi)//2
            with torch.no_grad():
                input_data[0][lens+5] = mid
                latency = self._predictor(input_data).numpy()[0]
            # logging.info("{} {} {} {} {}".format(qos, latency, lo, mid, hi))
            if qos < latency:
                hi = mid-1
            else:
                lo = mid
                if qos-latency < self._threashold or hi-lo <= 2:
                    break
        with torch.no_grad():
            input_data[0][lens+5] = lo
            self._predicted_latency = self._predictor(input_data).numpy()[0]
        return lo

    def Abacus_reset(self, search_times=0):
        # 写调度结果到文件
        if self._schedule_flag >= 0:
            # print(len(self._barrier))
            self._barrier[self._schedule_flag].wait()
            self._schedule_flag = -1
            for model_id in self._serve_combination:
                query: Query = self._scheduled_queries[model_id]
                if query is not None:
                    self._wr.writerow(
                        [
                            query.id,
                            query.model_id,
                            query.batch_size,
                            query.seq_len,
                            self._search_times,
                            query.load_id,
                            query.latency_ms(),
                        ]
                    )
                    self._scheduled_queries[model_id] = None
            self._search_times = search_times

    def FCFS_schedule(self, abandon=False):
        waiting_scheduling = {}
        abandoned_scheduling = []
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if abandon == True:
                if headroom > 0:
                    query.set_op_pos(-1)
                    arrival_stamp = query.start_stamp
                    waiting_scheduling[model_id] = arrival_stamp
                else:
                    abandoned_scheduling.append(model_id)
            else:
                query.set_op_pos(-1)
                arrival_stamp = query.start_stamp
                waiting_scheduling[model_id] = arrival_stamp

        # 按照到达时间进行排序
        waiting_scheduling = sorted(
            waiting_scheduling.items(), key=lambda x: x[1])

        # 按照waiting_scheduling中的顺序线性执行
        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                (
                    query.id,
                    "new",
                    0,
                    query.start_pos,
                    query.end_pos,
                    query.batch_size,
                    query.seq_len,
                )
            )
            self._barrier[0].wait()
            self.clean_scheduling_queries(model_ids=model_id)
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        query.load_id,
                        query.latency_ms(),
                    ]
                )
            )

        # abandoned直接drop
        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    def SJF_schedule(self, abandon=False):
        waiting_scheduling = {}
        abandoned_scheduling = []
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if abandon == True:
                if headroom > 0:
                    query.set_op_pos(-1)
                    with torch.no_grad():
                        latency = self._predictor(
                            self.get_layer_feature(
                                start_pos=0, end_pos=0, search_ways=1, qos_query=query
                            )[1]
                        )[0]
                    waiting_scheduling[model_id] = latency
                else:
                    abandoned_scheduling.append(model_id)
            else:
                query.set_op_pos(-1)
                with torch.no_grad():
                    latency = self._predictor(
                        self.get_layer_feature(
                            start_pos=0, end_pos=0, search_ways=1, qos_query=query
                        )[1]
                    )[0]
                waiting_scheduling[model_id] = latency

        # 按照预测的latency排序
        waiting_scheduling = sorted(
            waiting_scheduling.items(), key=lambda x: x[1])

        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                (
                    query.id,
                    "new",
                    0,
                    query.start_pos,
                    query.end_pos,
                    query.batch_size,
                    query.seq_len,
                )
            )
            self._barrier[0].wait()
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        query.load_id,
                        query.latency_ms(),
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    # earliest deadline
    def EDF_schedule(self, abandon=False):
        waiting_scheduling = {}
        abandoned_scheduling = []
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if abandon == True:
                if headroom > 0:
                    query.set_op_pos(-1)
                    waiting_scheduling[model_id] = headroom
                else:
                    abandoned_scheduling.append(model_id)
            else:
                query.set_op_pos(-1)
                waiting_scheduling[model_id] = headroom

        # 按照deadline，即headroom排序
        waiting_scheduling = sorted(
            waiting_scheduling.items(), key=lambda x: x[1])

        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                (
                    query.id,
                    "new",
                    0,
                    query.start_pos,
                    query.end_pos,
                    query.batch_size,
                    query.seq_len,
                )
            )
            self._barrier[0].wait()
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        query.load_id,
                        query.latency_ms(),
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        query.load_id,
                        -1,
                    ]
                )
            )
            self.clean_scheduling_queries(model_ids=model_id)
        self._result_file.flush()

    # 将query从请求队列self._queues[model_id]，添加到待调度队列self._scheduling_queries
    def pop_queries(self):
        # [0,1]
        for model_id in self._serve_combination:
            if (self._scheduling_queries[model_id] is None) and (
                not self._queues[model_id].empty()
            ):
                self._scheduling_queries[model_id] = self._queues[model_id].get(
                )

    # 待调度队列self._scheduling_queries是否非空
    def remain_schedule(self):
        for key in self._scheduling_queries:
            if self._scheduling_queries[key] is not None:
                return True
        return False

    # 清空_scheduling_queries队列
    def clean_scheduling_queries(self, model_ids):
        if isinstance(model_ids, list):
            for i in model_ids:
                self._scheduling_queries[i] = None
        else:
            self._scheduling_queries[model_ids] = None

    # 按照qos净空，headroom进行排序
    def urgent_sort(self, abandon):
        waiting_scheduling = {}
        abandoned_scheduling = []
        # [0,1]
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if abandon == True:
                if headroom > 0:
                    waiting_scheduling[model_id] = headroom
                else:
                    abandoned_scheduling.append(model_id)
            else:
                waiting_scheduling[model_id] = headroom
        # 按照qos净空，headroom进行排序
        waiting_scheduling = sorted(
            waiting_scheduling.items(), key=lambda x: x[1])
        return waiting_scheduling, abandoned_scheduling

    # 创建整个query的input features
    def get_query_feature(
        self,
        search_ways,
        qos_query: Query,
        l_query: Query = None,
        m_query: Query = None,
        r_query: Query = None,
    ):
        # 创建search_ways个self._models_feature类型的map
        input_feature = torch.zeros(search_ways, self._models_feature)
        lens = len(self._run_config.models_id)
        if self._run_config.predictor == "layer2":
            lens -= 1
        input_feature[0:search_ways, qos_query.model_id] += 1
        input_feature[0:search_ways, lens] = qos_query.start_pos
        input_feature[0:search_ways, lens+1] = qos_query.end_pos
        input_feature[0:search_ways, lens+2] = qos_query.batch_size
        input_feature[0:search_ways, lens+3] = qos_query.seq_len
        if l_query is not None:
            input_feature[1:search_ways, l_query.model_id] += 1
            input_feature[1:search_ways, lens+4] = l_query.end_pos
            input_feature[1:search_ways, lens+5] = l_query.op_len
            input_feature[1:search_ways, lens+6] = l_query.batch_size
            input_feature[1:search_ways, lens+7] = l_query.seq_len
        if m_query is not None:
            input_feature[2:search_ways, m_query.model_id] += 1
            input_feature[2:search_ways, lens+8] = m_query.end_pos
            input_feature[2:search_ways, lens+9] = m_query.op_len
            input_feature[2:search_ways, lens+10] = m_query.batch_size
            input_feature[2:search_ways, lens+11] = m_query.seq_len
        if r_query is not None:
            input_feature[3, r_query.model_id] += 1
            input_feature[3, lens+12] = r_query.end_pos
            input_feature[3, lens+13] = r_query.op_len
            input_feature[3, lens+14] = r_query.batch_size
            input_feature[3, lens+15] = r_query.seq_len
        return input_feature

    # search_ways表示有多少种可能
    # 1-search_ways每个分别往前推进step，即start-end，end是在变化的
    def get_layer_feature(
        self,
        start_pos,
        end_pos,
        search_ways,
        qos_query: Query,
        l_query: Query = None,
        m_query: Query = None,
        r_query: Query = None,
    ):

        lens = len(self._run_config.models_id)
        if self._run_config.predictor == "layer2":
            lens -= 1
        step = (end_pos - start_pos) // (search_ways + 1)
        poses = []
        # self._models_feature = len(Query.MODELS_LEN) + 4 * run_config.total_models
        # input_feature size为：search_ways × self._models_feature
        input_feature = torch.zeros(search_ways, self._models_feature)
        # 相应bitmap设置为1
        input_feature[0:search_ways, qos_query.model_id] += 1
        # model features
        input_feature[0:search_ways, lens] = qos_query.start_pos
        input_feature[0:search_ways, lens+1] = qos_query.end_pos
        input_feature[0:search_ways, lens+2] = qos_query.batch_size
        input_feature[0:search_ways, lens+3] = qos_query.seq_len

        # single_input = self.culops(
        #     self._run_config.models_name[qos_query.model_id], qos_query.start_pos, qos_query.end_pos, qos_query.batch_size)
        # items = torch.cat((torch.zeros(8), single_input))
        # input_feature[0:search_ways] = items

        if r_query is not None:
            input_feature[0:search_ways, l_query.model_id] += 1
            input_feature[0:search_ways, lens+4] = l_query.start_pos
            input_feature[0:search_ways, lens+5] = l_query.end_pos
            input_feature[0:search_ways, lens+6] = l_query.batch_size
            input_feature[0:search_ways, lens+7] = l_query.seq_len

            input_feature[0:search_ways, m_query.model_id] += 1
            input_feature[0:search_ways, lens+8] = m_query.start_pos
            input_feature[0:search_ways, lens+9] = m_query.end_pos
            input_feature[0:search_ways, lens+10] = m_query.batch_size
            input_feature[0:search_ways, lens+11] = m_query.seq_len

            input_feature[0:search_ways, r_query.model_id] += 1
            input_feature[0:search_ways, lens+12] = r_query.start_pos
            # TODO ?
            # input_feature[0:search_ways, 20] = r_query.end_pos
            input_feature[0:search_ways, lens+14] = r_query.batch_size
            input_feature[0:search_ways, lens+15] = r_query.seq_len

            for i in range(search_ways):
                start_pos += step
                # 每个search_ways start_pos不同
                input_feature[i, lens+13] = start_pos
                poses.append(start_pos)
        elif m_query is not None:
            input_feature[0:search_ways, l_query.model_id] += 1
            input_feature[0:search_ways, lens+4] = l_query.start_pos
            input_feature[0:search_ways, lens+5] = l_query.end_pos
            input_feature[0:search_ways, lens+6] = l_query.batch_size
            input_feature[0:search_ways, lens+7] = l_query.seq_len

            input_feature[0:search_ways, m_query.model_id] += 1
            input_feature[0:search_ways, lens+8] = m_query.start_pos
            # input_feature[0:search_ways, lens+9] = m_query.end_pos
            input_feature[0:search_ways, lens+10] = m_query.batch_size
            input_feature[0:search_ways, lens+11] = m_query.seq_len

            for i in range(search_ways):
                start_pos += step
                input_feature[i, lens+9] = start_pos
                poses.append(start_pos)
        # TODO ?
        if l_query is not None:
            input_feature[0:search_ways, l_query.model_id] += 1
            input_feature[0:search_ways, lens+4] = l_query.start_pos
            # input_feature[0:search_ways, lens+5] = l_query.end_pos
            input_feature[0:search_ways, lens+6] = l_query.batch_size
            input_feature[0:search_ways, lens+7] = l_query.seq_len

            for i in range(search_ways):
                start_pos += step
                input_feature[i, lens+5] = start_pos
                poses.append(start_pos)
        logging.debug("input feature: {}".format(input_feature))
        return poses, input_feature

    def parsejson(self, model_name):
        file_path = os.path.join(
            "/home/onceas/wanna/Abacus/data/op_details", model_name+".json")
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            return content

    def culops(self, model_name, start, end, batch):
        res = [0, 0, 0, 0, 0, 0, 0, batch]
        # print(model_name)
        for i in range(start-1, end):
            idx = 0
            for k in self.op_details[model_name][i].keys():
                v = self.op_details[model_name][i][k]
                res[idx] += v
                idx += 1
        return torch.FloatTensor(res)
        # return torch.FloatTensor(res).resize(1, len(res))
