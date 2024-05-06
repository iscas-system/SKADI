#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# %%
import csv
import torch.nn as nn
import pkg.modeling.predictor.mlp as mlp
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import os
import glob
import json
from pkg.modeling.utils import AverageMeter
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


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
    # "densenet121": 31,
    # "densenet161": 111,
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
    # 11: "densenet121",
    # 12: "densenet161",
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
    # "densenet121": 11,
    # "densenet161": 12,
}


def parsejson(model_name):
    file_path = os.path.join(
        "/home/onceas/wanna/mt-dnn/data/op_details", model_name+".json")
    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)
        return content


def culops(maps, start, end):
    res = [0, 0, 0, 0, 0, 0, 0, 0]
    # print(maps)
    for i in range(start-1, end):
        idx = 0
        for k in maps[i].keys():
            v = maps[i][k]
            res[idx] += v
            idx += 1
    return res


def get_feature_latency(data, models_id, total_models=2):

    n = data.shape[0]
    feature_data = []
    latency_data = []

    for i in range(n):
        line = data[i]
        # print(line)
        model_records = {}
        model_feature = np.zeros([len(models_id)])
        for i in range(total_models):
            model_id = models_id[line[i * 5]]
            model_feature[model_id] += 1
            model_record = np.array(
                [
                    int(line[1 + i * 5]),
                    int(line[2 + i * 5]),
                    int(line[3 + i * 5]),
                    int(line[4 + i * 5]),
                ]
            )
            model_records[model_id] = model_record
        # print(model_records)
        model_records = dict(sorted(model_records.items()))
        # print(model_records)
        for key in model_records:
            model_feature = np.concatenate((model_feature, model_records[key]))

        # print(model_feature)
        # print(model_feature)
        feature_data.append(model_feature)
        # print(model_feature)
        # print(line[-3])
        latency_data.append(float(line[-3]))
    feature_data = np.array(feature_data)
    latency_data = np.array(latency_data)
    return feature_data, latency_data


def get_feature_latency2(data, models_id, need_name, total_models=2):

    print(models_id)
    print("need_name:", need_name)
    n = data.shape[0]
    feature_data = []
    latency_data = []

    details = {}
    for name in need_name:
        # print(id)
        maps = parsejson(name)
        details[models_id[name]] = maps

    for i in range(n):
        line = data[i]
        # print(line)
        model_records = {}
        # model_feature = np.zeros([len(models_id)])
        model_feature = np.zeros(0)
        for i in range(total_models):
            model_id = models_id[line[i * 5]]
            # model_feature[model_id] += 1
            model_record = np.array(
                [
                    int(line[1 + i * 5]),
                    int(line[2 + i * 5]),
                    int(line[3 + i * 5]),
                    int(line[4 + i * 5]),
                ]
            )
            items = culops(details[model_id], model_record[0], model_record[1])
            # model_records[model_id] = model_record
            items[-1] = model_record[2]
            model_records[model_id] = items
        # print(model_records)
        # print("before:", model_records)
        #
        model_records = dict(sorted(model_records.items(),
                             key=lambda item: models_sz[models_reid[item[0]]]))
        # model_records = dict(sorted(model_records.items()))
        # print("after:", model_records)
        # print(model_records)
        for key in model_records:
            model_feature = np.concatenate((model_feature, model_records[key]))

        # print(model_feature)
        feature_data.append(model_feature)
        # print(model_feature)
        # print(line[-3])
        latency_data.append(float(line[-3]))
    feature_data = np.array(feature_data)
    latency_data = np.array(latency_data)
    return feature_data, latency_data


def load_single_file(predictor_type, filepath, models_id, total_models=2):
    print("load file:", filepath)
    filename = filepath.split('/')[-1].split('.')[0]
    tmp = [name for name in filename.split('_')]
    need_name = []
    for name in tmp:
        if name == 'v3':
            continue
        elif name == "inception":
            need_name.append("inception_v3")
        else:
            need_name.append(name)

    # TODO 此处只算成对的，忽略单个的
    if len(need_name) == 1:
        return None, None
    data = pd.read_csv(filepath, header=0)
    data = data.values.tolist()
    total_data_num = len(data)
    print("{} samples loaded from {}".format(total_data_num, filepath))
    data = np.array(data)
    # TODO fix predictor
    if predictor_type == "layer":
        return get_feature_latency(data, models_id, total_models)
    else:
        return get_feature_latency2(data, models_id, need_name, total_models)


def load_torch_data(
    model_combinatin,
    batch_size,
    train_ratio,
    models_id,
    data_path="/home/onceas/wanna/mt-dnn/data/profiling",
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


def load_data_for_sklearn(
    model_combinatin,
    split_ratio,
    models_id,
    data_path="/home/cwh/Lego/data",
    total_models=2,
    predictor_type=None,
):
    all_feature = None
    all_latency = None
    if model_combinatin == "all":
        for filename in glob.glob(os.path.join(data_path, "*.csv")):
            feature_data, latency_data = load_single_file(
                predictor_type, filename, models_id, total_models
            )
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
    X, y = (all_feature, all_latency)
    y = y / 1000
    data = np.concatenate((X, np.reshape(y, (-1, 1))), axis=1)
    np.random.shuffle(data)
    n = data.shape[0]
    split = int(split_ratio * n)
    train = data[:split, :]
    test = data[split:, :]
    trainX = train[:, :-1]
    trainY = train[:, -1:].reshape(-1)
    testX = test[:, :-1]
    testY = test[:, -1:].reshape(-1)

    return trainX, trainY, testX, testY


# %%

def load_model(path, predictor_type):
    first_layer = None
    if predictor_type == "layer":
        first_layer = 19
    else:
        first_layer = 16
    model = mlp.MLPregression(first_layer).to('cpu')
    weights = torch.load(path, map_location='cpu')
    model.load_state_dict(weights)
    return model


def save_result_file(result_file, combination, mae, mape):
    result_file = open(result_file, "a+")
    wr = csv.writer(result_file, dialect="excel")
    if combination is None:
        combination = "all"
    wr.writerow([combination, mae, mape])


def validate(model, validate_pair, _test_loader, save_result=False, save_model=False, perf=False):
    origin_latency = None
    predict_latency = None
    total_batches = len(_test_loader)
    test_loss = AverageMeter()
    _loss_func = nn.MSELoss()
    with torch.no_grad():
        with tqdm(total=total_batches, desc="Testing ") as t:
            for step, (input, latency) in enumerate(_test_loader):
                input, latency = input.to(
                    'cpu'), latency.to('cpu')
                pre_y = model(input)
                loss = _loss_func(pre_y, latency)
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
    print("{} mae: {}, mape: {}".format(validate_pair, mae, mape))

    if save_result is True:
        validate_path = "/home/onceas/wanna/mt-dnn/result/2080Ti/validate"
        if not os.path.exists(validate_path):
            os.makedirs(validate_path)
        result_file = os.path.join(validate_path, "result.csv")
        save_result_file(result_file, validate_pair, mae, mape)
        index = np.argsort(origin_latency)
        plt.figure(figsize=(12, 5))
        plt.plot(
            np.arange(len(origin_latency)),
            origin_latency[index],
            "r",
            label="actual time",
        )
        plt.scatter(
            np.arange(len(predict_latency)),
            predict_latency[index],
            s=3,
            c="b",
            label="prediction time",
        )
        # plt.title("existing work:"+validate_pair+" validate")
        plt.title("our work:"+validate_pair+" validate")
        plt.legend(loc="upper left")
        plt.grid()
        plt.xlabel("index")
        plt.ylabel("execution time(ms)")

        plt.savefig(
            os.path.join(validate_path,
                         validate_pair + "_test.pdf"),
            bbox_inches="tight",
        )


if __name__ == "__main__":

    # validate_pair = "resnet101_inception_v3"
    # validate_pair = "resnet50_vgg19"
    dir_path = "/home/onceas/wanna/mt-dnn/data/profile/2080Ti/2in7-vgg13-batch16"
    predictor_type = "operator"
    # predictor_type = "layer"
    model = load_model(
        "/home/onceas/wanna/mt-dnn/model/2080Ti/2in7/all-"+predictor_type+".ckpt", predictor_type)
    print(model)
    # model = load_model("/home/onceas/wanna/mt-dnn/model/2080Ti/2in7/all-layer.ckpt")
    model.eval()
    file_list = os.listdir(dir_path)
    for file in file_list:
        validate_pair = file.split(".")[0]
        train_dataloader, test_dataloader = load_torch_data(
            validate_pair,
            128,
            0,
            models_id,
            data_path=dir_path,
            total_models=2,
            predictor_type=predictor_type,
        )
        print(len(test_dataloader.dataset))
        print(test_dataloader.dataset[0])
        validate(model, validate_pair, test_dataloader, True)


# %%
