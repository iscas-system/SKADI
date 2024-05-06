#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# cluster client for Abacus


import logging
import random
import time

import torch.multiprocessing as mp
from torch.multiprocessing import Process

from pkg.utils import Query
from pkg.option import RunConfig
from pkg.loadbalancer.abacus import AbacusLoadBalancer
from pkg.loadbalancer.clock import ClockLoadBalancer
import pkg.loadbalance_pb2 as loadbalancer_pb2
import pkg.loadbalance_pb2_grpc as loadbalancer_pb2_grpc
import pkg.service_pb2 as service_pb2
import pkg.service_pb2_grpc as service_pb2_grpc
import grpc


class ClusterClient():
    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._queues = {}
        self._load_balancers = {}
        self._qos_target = run_config.qos_target
        self._lb_ip = run_config.lb_ip
        channel = grpc.insecure_channel("{}:50052".format(self._lb_ip))
        stub = loadbalancer_pb2_grpc.DNNLoadBalancerStub(channel=channel)
        self._grpc_stub = stub
        random.seed(0)

    def start_long_term_test(self):
        logging.info("warm up all servers")
        for i in range(10):
            id = 0
            model_id = random.choice(self._run_config.serve_combination)
            sleep_duration = random.expovariate(500)
            bs = random.choice(self._run_config.supported_batchsize)
            # seq_len = (
            #     random.choice(
            #         self._run_config.supported_seqlen) if model_id == 6 else 0
            # )
            seq_len = 0
            self.send_query(
                id=id, model_id=model_id, batch_size=bs, seq_len=seq_len, load_id=-1
            )
            time.sleep(sleep_duration)
        logging.info("warmed up all servers")
        id = 0
        load_id = 0
        total_loads = len(self._run_config.loads)
        average_duration = self._run_config.loads[load_id]/16/6
        start_stamp = time.time()
        while True:
            if (time.time() - start_stamp) >= self._run_config.load_change_dura:
                load_id += 1
                logging.info("load changed: {}".format(load_id))
                if load_id == total_loads:
                    break
                average_duration = self._run_config.loads[load_id] / 16/6
                start_stamp = time.time()
            id += 1
            model_id = random.choice(self._run_config.serve_combination)
            sleep_duration = random.expovariate(average_duration)
            logging.debug("{}".format(sleep_duration))
            bs = random.choice(self._run_config.supported_batchsize)
            # seq_len = (
            #     random.choice(
            #         self._run_config.supported_seqlen) if model_id == 6 else 0
            # )
            seq_len = 0
            self.send_query(
                id=id,
                model_id=model_id,
                batch_size=bs,
                seq_len=seq_len,
                load_id=load_id,
            )
            time.sleep(sleep_duration)

    def send_query(self, id, model_id, batch_size, seq_len, load_id):
        req = loadbalancer_pb2.Query(
            id=id,
            model_id=model_id,
            bs=batch_size,
            seq_len=seq_len,
            start_stamp=time.time(),
            qos_target=self._qos_target,
            load_id=load_id,
        )
        return self._grpc_stub.LBInference(req)
