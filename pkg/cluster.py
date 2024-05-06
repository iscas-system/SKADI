#!/usr/bin/env python3
# -*- coding:utf-8 -*-

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
from pkg.modeling.dataloader import load_model, culops, parsejson
import pandas as pd
import os
import json
import numpy as np
import torch
from pkg.statistics import merge_abacus, abacus_preprocess, get_abacus_latency
from google.protobuf.json_format import MessageToJson, MessageToDict

models_sz = {
    "resnet50": 98,
    "resnet101": 171,
    "resnet152": 231,
    "inception_v3": 104,
    "vgg16": 528,
    "vgg19": 549,
    "resnet18": 45,
    "resnet34": 84,
    "googlenet": 50,
    "vgg11": 507,
    "vgg13": 508,
}

models_reid = {
    0: "resnet50",
    1: "resnet101",
    2: "resnet152",
    3: "inception_v3",
    4: "vgg16",
    5: "vgg19",
    6: "resnet18",
    7: "resnet34",
    8: "googlenet",
    9: "vgg11",
    10: "vgg13",
}

models_id = {
    "resnet50": 0,
    "resnet101": 1,
    "resnet152": 2,
    "inception_v3": 3,
    "vgg16": 4,
    "vgg19": 5,
    "resnet18": 6,
    "resnet34": 7,
    "googlenet": 7,
    "vgg11": 8,
    "vgg13": 10,
}

qos_target = {
    "resnet50resnet101": 100,
    "resnet50resnet152": 150,
    "resnet50inception_v3": 100,
    "resnet50vgg16": 50,
    "resnet50vgg19": 50,
    "resnet50bert": 75,
    "resnet101resnet152": 160,
    "resnet101inception_v3": 130,
    "resnet101vgg16": 90,
    "resnet101vgg19": 80,
    "resnet101bert": 130,
    "resnet152inception_v3": 150,
    "resnet152vgg16": 150,
    "resnet152vgg19": 150,
    "resnet152bert": 150,
    "inception_v3vgg16": 80,
    "inception_v3vgg19": 90,
    "inception_v3bert": 80,
    "vgg16vgg19": 90,
    "vgg16bert": 60,
    "vgg19bert": 60,
    "vgg16resnet34": 40,
    "vgg19resnet34": 40,
}


class Cluster(loadbalancer_pb2_grpc.DNNLoadBalancerServicer):
    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._queues = {}
        self._load_balancers = {}
        self._qos_target = run_config.qos_target
        random.seed(0)
        if run_config.policy != "loadbalancer":
            self.start_load_balancer()
        logging.info("Loadbalancer Initialized")

    def start_load_balancer(self):
        logging.info("Model queues Initializing")
        for model_id in self._run_config.serve_combination:
            self._queues[model_id] = mp.Queue()
        logging.info("Loadbalancer Initializing")

        # 每种类型的模型都有一个负载均衡器
        if self._run_config.policy == "Abacus":
            for model_id in self._run_config.serve_combination:
                load_balancer = AbacusLoadBalancer(
                    run_config=self._run_config,
                    model_id=model_id,
                    query_q=self._queues[model_id],
                    qos_target=self._run_config.qos_target,
                )
                load_balancer.start()
                self._load_balancers[model_id] = load_balancer
        elif self._run_config.policy == "Clock":
            self._node_q = mp.Queue()
            for node_id in range(self._run_config.node_cnt):
                self._node_q.put(node_id)
            balancer_id = 0
            for model_id in self._run_config.serve_combination:
                logging.info(
                    "Initializing Clock Loadbalancer for model {}".format(
                        model_id)
                )
                # for _ in range(4):
                #     self._load_balancers[balancer_id] = ClockLoadBalancer(
                #         loader_id=balancer_id,
                #         run_config=self._run_config,
                #         model_id=model_id,
                #         query_q=self._queues[model_id],
                #         node_q=self._node_q,
                #         qos_target=self._run_config.qos_target,
                #     )
                #     self._load_balancers[balancer_id].start()
                #     balancer_id += 1
                self._load_balancers[balancer_id] = ClockLoadBalancer(
                    loader_id=balancer_id,
                    run_config=self._run_config,
                    model_id=model_id,
                    query_q=self._queues[model_id],
                    node_q=self._node_q,
                    qos_target=self._run_config.qos_target,
                )
                self._load_balancers[balancer_id].start()
                balancer_id += 1
        else:
            raise NotImplementedError
        # logging.info("sleep for 5 seconds and start testing")

    # grpc LBInference
    def LBInference(self, request, context):
        query_id = request.id
        model_id = request.model_id
        bs = request.bs
        seq_len = request.seq_len
        start_stamp = request.start_stamp
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
        # logging.info("loadbalancer received query: {}".format(query))
        return loadbalancer_pb2.Result(node_id=0, accepted=True)

    def GetLayers(self, request, context):
        import os
        import json

        start = request.start-1
        end = request.end
        model = request.model
        base_dir = "/home/onceas/wanna/mt-dnn/data/op_details"
        filepath = os.path.join(base_dir, model+".json")
        with open(filepath, 'r') as file:
            data = json.load(file)
        resp = loadbalancer_pb2.GetLayersResponse(code=200, message="ok")
        for i in range(start, end):
            line = data[i]
            # print(type(line))
            # print(line)
            layers = loadbalancer_pb2.Layers()
            for key in line.keys():
                layer = loadbalancer_pb2.Layer(name=key, num=line[key])
                layers.layers.append(layer)
            resp.layers.append(layers)
        return resp

    def load_single_file(self, filepath, start1, end1, bs1, start2, end2, bs2):
        print("load file:", filepath)
        data = pd.read_csv(filepath, header=0)
        data = data.values.tolist()
        total_data_num = len(data)
        print("{} samples loaded from {}".format(total_data_num, filepath))
        data = np.array(data)
        for line in data:
            if line[1] == start1 and line[2] == end1 and line[6] == start2 and line[7] == end2 and line[3] == bs1 and line[8] == bs2:
                return float(line[-2])/1000
        return 0

    def GetPrediction(self, request, context):

        start1 = request.model1.start-1
        end1 = request.model1.end
        model1 = request.model1.model
        bs1 = request.model1.bs
        start2 = request.model2.start-1
        end2 = request.model2.end
        model2 = request.model2.model
        bs2 = request.model2.bs
        base_dir = "/home/onceas/wanna/mt-dnn/data/profile/2080Ti/2in7"

        predictor_type = "operator"
        # predictor_type = "layer"
        model = load_model(
            "/home/onceas/wanna/mt-dnn/model/2080Ti/2in7/all-"+predictor_type+".ckpt", predictor_type)
        model.eval()

        model_records = dict()

        maps = parsejson(model1)
        items = culops(maps, start1+1, end1)
        items[-1] = bs1
        model_records[models_id[model1]] = items

        maps = parsejson(model2)
        items = culops(maps, start2+1, end2)
        items[-1] = bs2
        model_records[models_id[model2]] = items

        model_records = dict(sorted(model_records.items(),
                                    key=lambda item: models_sz[models_reid[item[0]]]))

        model_feature = np.zeros(0)
        for key in model_records:
            model_feature = np.concatenate((model_feature, model_records[key]))

        resp = loadbalancer_pb2.GetPredictionResponse(code=200, message="ok")
        input = np.array([model_feature])
        input = torch.from_numpy(input.astype(np.float32))
        print(input.shape)
        print(input)
        print(model(input))
        resp.prediction = model(input).detach().numpy()[0]
        resp.actual = self.load_single_file(os.path.join(
            base_dir, model1+"_"+model2+".csv"), str(start1), str(end1), str(bs1), str(start2), str(end2), str(bs2))
        return resp

    def load_result_file(self, filepath):
        data = pd.read_csv(filepath, usecols=[0, 13, 14, 15], header=0)
        data = data.values.tolist()
        return data

    def GetResults(self, request, context):
        query = request.query
        load = request.load
        deadline = request.deadline
        model1 = request.model1
        model2 = request.model2

        project_path = "/home/onceas/wanna/mt-dnn/results/cluster-0.80/"
        abacus_src_path = os.path.join(project_path, "Abacus")
        abacus_raw_file = os.path.join(
            project_path, "Abacus_raw.csv")
        abacus_clean_file = os.path.join(
            project_path, "Abacus_clean.csv")
        result_file = os.path.join(
            project_path, "results.json")
        if os.path.exists(result_file):
            with open(result_file, 'r') as file:
                data = json.load(file)
                resp = loadbalancer_pb2.GetResultsResponse(
                    code=200, message="ok")
                # print(data)
                # print(type(data))
                resp.violation1.extend(data["violation1"])
                resp.throughout1.extend(data["throughout1"])
                resp.tail1.extend(data["tail1"])
                resp.mean1.extend(data["mean1"])
                resp.violation2.extend(data["violation2"])
                resp.throughout2.extend(data["throughout2"])
                resp.tail2.extend(data["tail2"])
                resp.mean2.extend(data["mean2"])
        else:
            merge_abacus(src_path=abacus_src_path, dst_file=abacus_raw_file)

            abacus_preprocess(abacus_raw_file, abacus_clean_file)

            model1_data, model2_data = get_abacus_latency(abacus_clean_file)
            model1_99 = model1_data[:30, 0]
            model1_mean = model1_data[:30, 1]
            model1_load = model1_data[:30, 2]
            model1_vio = model1_data[:30, 3]

            model2_99 = model2_data[:30, 0]
            model2_mean = model2_data[:30, 1]
            model2_load = model2_data[:30, 2]
            model2_vio = model2_data[:30, 3]

            resp = loadbalancer_pb2.GetResultsResponse(code=200, message="ok")
            resp.violation1.extend(model1_vio)
            resp.throughout1.extend(model1_load)
            resp.tail1.extend(model1_99)
            resp.mean1.extend(model1_mean)

            resp.violation2.extend(model2_vio)
            resp.throughout2.extend(model2_load)
            resp.tail2.extend(model2_99)
            resp.mean2.extend(model2_mean)

            with open(result_file, 'w') as f:
                json.dump(MessageToDict(resp), f)
        return resp
