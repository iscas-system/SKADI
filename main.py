#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import logging
import sys
import grpc
from concurrent import futures
import torch
import torch.multiprocessing as mp

# from pkg.profiler import profile
from pkg.profiler2 import profile
from pkg.server import AbacusServer, ClockServer
from pkg.cluster import Cluster
from pkg.cluster_client import ClusterClient
from pkg.trainer import train_predictor
from pkg.option import parse_options
from pkg.background import background
import pkg.service_pb2 as service_pb2
import pkg.service_pb2_grpc as service_pb2_grpc
import pkg.loadbalance_pb2 as loadbalancer_pb2
import pkg.loadbalance_pb2_grpc as loadbalancer_pb2_grpc

import torch.autograd.profiler as profiler

if __name__ == "__main__":
    # with profiler.profile(use_cuda=True,  profile_memory=True) as prof:
    run_config = parse_options()
    # mp.set_start_method("spawn")
    torch.set_printoptions(linewidth=200)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if run_config.debug is True else logging.INFO,
        format='%(asctime)s   %(levelname)s   %(message)s'
    )
    # print(run_config)
    if run_config.task == "profile":
        profile(run_config=run_config)
    elif run_config.task == "server":
        if run_config.platform == "single":
            server = AbacusServer(run_config=run_config)
            server.start_test()
            server.stop_test()
        elif run_config.platform == "cluster":
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
            if run_config.policy == "Abacus":
                ServerClass = AbacusServer
            elif run_config.policy == "Clock":
                ServerClass = ClockServer
            else:
                raise NotImplementedError
            service_pb2_grpc.add_DNNServerServicer_to_server(
                ServerClass(run_config=run_config), server
            )
            server.add_insecure_port("[::]:50051")
            server.start()
            server.wait_for_termination()
        else:
            raise NotImplementedError
    elif run_config.task == "loadbalancer":
        # add loadbalancer server
        lbserver = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        loadbalancer_pb2_grpc.add_DNNLoadBalancerServicer_to_server(
            Cluster(run_config=run_config), lbserver
        )
        lbserver.add_insecure_port("[::]:50052")
        lbserver.start()
        lbserver.wait_for_termination()
    elif run_config.task == "scheduler":
        abacus_client = ClusterClient(run_config=run_config)
        abacus_client.start_long_term_test()
    elif run_config.task == "train":
        train_predictor(run_config)
    elif run_config.task == "background":
        background(args=run_config)
