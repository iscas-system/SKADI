#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import psutil
import os
import time
import math
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from pkg.modeling.predictor import LatencyPredictor
# from pkg.modeling.dataloader import load_torch_data
from pkg.modeling.utils import AverageMeter
from pkg.option import RunConfig


def load_torch_data(
    model_combinatin,
    batch_size,
    train_ratio,
    models_id,
    data_path="/home/onceas/wanna/Abacus/data/profiling",
    total_models=2,
    predictor_type="layer",
):
    all_feature = None
    all_latency = None
    if model_combinatin == "all":
        for filename in glob.glob(os.path.join(data_path, "*.csv")):
            # print(filename)
            feature_data, latency_data = load_single_file(
                predictor_type, filename, models_id, total_models
            )
            if feature_data is None or latency_data is None:
                continue
            if all_feature is None or all_latency is None:
                all_feature = feature_data
                all_latency = latency_data
            else:
                all_feature = np.concatenate(
                    (all_feature, feature_data), axis=0)
                all_latency = np.concatenate(
                    (all_latency, latency_data), axis=0)

    else:
        filename = model_combinatin + ".csv"
        all_feature, all_latency = load_single_file(
            predictor_type, os.path.join(
                data_path, filename), models_id, total_models
        )

    feature_len = len(all_feature)
    latency_len = len(all_latency)
    assert feature_len == latency_len
    train_len = int(feature_len * train_ratio)
    test_len = feature_len - train_len
    all_feature = torch.from_numpy(all_feature.astype(np.float32))
    all_latency = all_latency / 1000
    all_latency = torch.from_numpy(all_latency.astype(np.float32))

    all_dataset = Data.TensorDataset(all_feature, all_latency)
    train_dataset, test_dataset = Data.random_split(
        all_dataset, [train_len, test_len])
    train_dataloader = None
    if train_len != 0:
        train_dataloader = Data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
    test_dataloader = Data.DataLoader(
        dataset=test_dataset, batch_size=128, shuffle=True
    )
    return train_dataloader, test_dataloader


class MLPregression(nn.Module):
    def __init__(self, first_layer=14):
        super(MLPregression, self).__init__()
        # print("first=", first_layer)
        self.hidden1 = nn.Linear(
            in_features=first_layer, out_features=32, bias=True)
        self.hidden2 = nn.Linear(32, 32)
        self.hidden3 = nn.Linear(32, 32)
        self.predict = nn.Linear(32, 1)

        # print(first_layer)
        # self.hidden1 = nn.Linear(
        #     in_features=first_layer, out_features=32, bias=True)
        # self.hidden2 = nn.Linear(32, 64)
        # self.hidden3 = nn.Linear(64, 64)
        # self.hidden4 = nn.Linear(64, 32)
        # self.predict = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        output = self.predict(x)
        return output[:, 0]


class MLPPredictor(LatencyPredictor):
    def __init__(
        self,
        run_config: RunConfig,
        models_id,
        epoch=30,
        batch_size=16,
        lr=0.001,
        lr_schedule_type="cosine",
        data_fname="all",
        split_ratio=0.8,
        path="/home/onceas/wanna/mt-dnn",
        device=0,
        total_models=2,
        mig=0,
    ):
        super().__init__(
            run_config,
            "mlp",
            models_id,
            epoch,
            batch_size,
            data_fname,
            split_ratio,
            path,
            total_models,
            mig,
        )
        self._train_loader, self._test_loader = load_torch_data(
            self._data_fname,
            self._batch_size,
            self._split_ratio,
            self._models_id,
            self._data_path,
            self._total_models,
            self._run_config.predictor,
        )
        self._total_batches = len(self._train_loader)
        self._run_config = run_config
        self._init_lr = lr
        self._lr_schedule_type = lr_schedule_type
        self._device = torch.device("cuda:{}".format(device))
        # TODO fix predictor input
        if self._run_config.predictor == "layer":
            self._firt_layer = self._total_models * \
                4 + len(self._run_config.models_len)
        else:
            self._firt_layer = self._total_models * 8

        self._model = MLPregression(self._firt_layer).to(self._device)
        print(self._model)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(), lr=self._init_lr)
        self._loss_func = nn.MSELoss()

    def train(self, save_result=False, save_model=False, perf=False):

        train_loss_all = []
        for epoch in range(self._total_epochs):
            self._model.train()

            print("total train data: {}".format(self._total_batches))
            train_loss = AverageMeter()

            with tqdm(
                total=self._total_batches, desc="Train Epoch #{}".format(
                    epoch + 1)
            ) as t:
                for i, (input, latency) in enumerate(self._train_loader):
                    print(input)
                    new_lr = self.adjust_learning_rate(epoch, i)
                    input, latency = input.to(
                        self._device), latency.to(self._device)
                    # print(input.shape)
                    # print(input)
                    output = self._model(input)
                    loss = self._loss_func(output, latency)
                    self._model.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    train_loss.update(loss.item(), input.size(0))
                    t.set_postfix(
                        {
                            "loss": train_loss.avg,
                            "input size": input.size(1),
                            "lr": new_lr,
                        }
                    )
                    t.update(1)
            self.validate()
            train_loss_all.append(train_loss.avg)

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_all, "bo-", label="Train loss")
        plt.legend()
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(
            os.path.join(self._result_path, self._data_fname + "_train.pdf"),
            bbox_inches="tight",
        )
        # plt.show()
        self.validate(save_result=save_result,
                      save_model=save_model, perf=perf)

    def validate(self, save_result=False, save_model=False, perf=False):
        origin_latency = None
        predict_latency = None
        self._model.eval()
        total_batches = len(self._test_loader)
        test_loss = AverageMeter()
        with torch.no_grad():
            with tqdm(total=total_batches, desc="Testing ") as t:
                for step, (input, latency) in enumerate(self._test_loader):
                    input, latency = input.to(
                        self._device), latency.to(self._device)
                    pre_y = self._model(input)
                    loss = self._loss_func(pre_y, latency)
                    test_loss.update(loss.item(), input.size(0))
                    t.set_postfix(
                        {
                            "loss": test_loss.avg,
                            "input_size": input.size(1),
                        }
                    )
                    t.update(1)

                    if origin_latency is None or predict_latency is None:
                        origin_latency = latency.cpu().data.numpy()
                        predict_latency = pre_y.cpu().data.numpy()
                    else:
                        origin_latency = np.concatenate(
                            (origin_latency, latency.cpu().data.numpy())
                        )
                        predict_latency = np.concatenate(
                            (predict_latency, pre_y.cpu().data.numpy())
                        )

        mae = np.average(np.abs(predict_latency - origin_latency))
        mape = np.average(
            np.abs(predict_latency - origin_latency) / origin_latency)
        print("mae: {}, mape: {}".format(mae, mape))

        if save_result is True:
            self.save_result(self._data_fname, mae, mape)
            index = np.argsort(origin_latency)
            plt.figure(figsize=(12, 5))
            plt.plot(
                np.arange(len(origin_latency)),
                origin_latency[index],
                "r",
                label="original y",
            )
            plt.scatter(
                np.arange(len(predict_latency)),
                predict_latency[index],
                s=3,
                c="b",
                label="prediction",
            )
            plt.legend(loc="upper left")
            plt.grid()
            plt.xlabel("index")
            plt.ylabel("y")
            plt.savefig(
                os.path.join(self._result_path,
                             self._data_fname + "_test.pdf"),
                bbox_inches="tight",
            )

        if save_model:
            self.save_model()

        # 设置不同数目的cores
        if perf is True:
            self._model.cpu().eval()
            total_cores = os.cpu_count()
            for cores in range(1, total_cores + 1):
                torch.set_num_threads(cores)
                # torch.set_num_interop_threads(cores)
                with torch.no_grad():
                    for bs in range(1, 17):
                        test_input = torch.rand((bs, self._firt_layer))
                        start_time = time.time()
                        for i in range(1000):
                            test_output = self._model(test_input)
                            # print(test_output[0])
                        end_time = time.time() - start_time
                        print(
                            "cores: {}, batch size: {}, Inference time: {} ms".format(
                                cores, bs, end_time
                            )
                        )

    def calc_learning_rate(self, epoch, batch=0):
        if self._lr_schedule_type == "cosine":
            T_total = self._total_epochs * self._total_batches
            T_cur = epoch * self._total_batches + batch
            lr = 0.5 * self._init_lr * \
                (1 + math.cos(math.pi * T_cur / T_total))
        elif self._lr_schedule_type is None:
            lr = self._init_lr
        else:
            raise ValueError("do not support: %s" % self._lr_schedule_type)
        return lr

    def adjust_learning_rate(self, epoch, batch=0):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self.calc_learning_rate(epoch, batch)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def save_model(self):
        print("saving model......")
        print(self._save_path)
        torch.save(
            self._model.state_dict(),
            os.path.join(self._save_path, self._data_fname+"-" +
                         self._run_config.predictor + ".ckpt"),
        )


def set_affinity(pid, core):
    p = psutil.Process(pid)
    p.cpu_affinity([core])


if __name__ == '__main__':
    # 获取系统CPU核心数量
    cpu_count = psutil.cpu_count()
    print(cpu_count)

    pid = os.getpid()
    set_affinity(pid, 10)

    # 获取当前程序的CPU绑定情况
    current_affinity = psutil.Process(pid).cpu_affinity()
    print("当前进程绑定的CPU核心：", current_affinity)

    predictor = MLPregression(16)
    predictor.load_state_dict(
        torch.load(
            "/home/onceas/wanna/mt-dnn/model/2080Ti/2in7/all-operator.ckpt", map_location="cpu")
    )
    predictor.eval()

    # bss = [1, 2, 4, 8, 16, 32]
    bss = [16]*100
    for bs in reversed(bss):
        # for bs in bss:
        print(bs)
        predictor(torch.rand(bs, 16))
        predictor(torch.rand(bs, 16))
        predictor(torch.rand(bs, 16))
        times = []
        for i in range(10):
            start = time.time()
            predictor(torch.rand(bs, 16))
            times.append(1000*(time.time()-start))
        # print(times)
        print(np.array(times).mean())

    # 0.099ms
    # 0.106ms
    # 0.107ms
    # 0.113ms
    # 0.122ms
