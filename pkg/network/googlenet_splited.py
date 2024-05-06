
# %%

# https://blog.csdn.net/qq_53144843/article/details/133279746

from torch import Tensor
from typing import Any, Callable, List, Optional, Tuple
from functools import partial
from collections import namedtuple
import warnings
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import os
from pkg.network.submod import *

model_urls = {
    "googlenet": "https://download.pytorch.org/models/googlenet-1378be20.pth",
}

model_dir = "/home/onceas/wanna/mt-dnn/model/plain"

__all__ = ["GoogLeNet", "GoogLeNetOutputs",
           "_GoogLeNetOutputs", "GoogLeNet_Weights", "googlenet"]

GoogLeNetOutputs = namedtuple(
    "GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {
    "logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


class GoogLeNet(nn.Module):
    __constants__ = ["aux_logits", "transform_input"]

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        transform_input: bool = False,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., nn.Module]]] = None,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
    ) -> None:
        super(GoogLeNet, self).__init__()
        self.moduleList = []

        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of GoogleNet will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        if len(blocks) != 3:
            raise ValueError(
                f"blocks length should be 3 instead of {len(blocks)}")
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        # 16+1+2（aux）
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(
                512, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(
                528, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # torch.flatten(x, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.trunc_normal_(
                        m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # print(type(self.modules()))
        pre = None
        for m in self.modules():
            if self.check_type(m):
                if isinstance(pre, nn.MaxPool2d):
                    # print(type(pre))
                    self.moduleList.append(nn.Sequential(pre))
                # print(type(m))
                self.moduleList.append(nn.Sequential(m))
            elif (isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Conv2d)) and len(self.moduleList) < 4:
                # print(type(m))
                self.moduleList.append(nn.Sequential(m))
            pre = m

        self.moduleList.append(nn.Sequential(avgpool_dropout_fc(
            self.avgpool, self.dropout, self.fc)))

        print(len(self.moduleList))
        # print(self.moduleList)
        # 解析每层的算子
        self.parse_ops(self.moduleList)
        print(len(self.layer2op))
        print(self.layer2op)

    def judge_single_op(self, seq, maps):
        if isinstance(seq, nn.Conv2d):
            maps["Conv2d"] += 1
        elif isinstance(seq, nn.BatchNorm2d):
            maps["BatchNorm2d"] += 1
        elif isinstance(seq, nn.ReLU):
            maps["ReLU"] += 1
        elif isinstance(seq, nn.MaxPool2d):
            maps["MaxPool2d"] += 1
        elif isinstance(seq, nn.Dropout):
            maps["Dropout"] += 1
        elif isinstance(seq, nn.AdaptiveAvgPool2d):
            maps["AdaptiveAvgPool2d"] += 1
        elif isinstance(seq, nn.Linear):
            maps["Linear"] += 1
        else:
            return None
        return maps

    def judge_group_ops(self, seq, maps):
        if isinstance(seq, Inception):
            maps["Conv2d"] += 6
            maps["BatchNorm2d"] += 6
            maps["ReLU"] += 6
            maps["MaxPool2d"] += 1
        elif isinstance(seq, InceptionAux):
            maps["Conv2d"] += 1
            maps["BatchNorm2d"] += 1
            maps["ReLU"] += 2
            maps["Linear"] += 2
            maps["Dropout"] += 1
        elif isinstance(seq, BasicConv2d):
            maps["Conv2d"] += 1
            maps["BatchNorm2d"] += 1
            maps["ReLU"] += 1
        else:
            return None
        return maps

    def parse_ops(self, layers):
        self.layer2op = []
        for layer in layers:
            # print("layer type ", type(layer))
            maps = {
                "Conv2d": 0,
                "BatchNorm2d": 0,
                "ReLU": 0,
                "MaxPool2d": 0,
                "AdaptiveAvgPool2d": 0,
                "Linear": 0,
                "Dropout": 0
            }
            if isinstance(layer, nn.Sequential):
                for seq in layer:
                    if isinstance(seq, Inception) or isinstance(seq, InceptionAux) or isinstance(seq, BasicConv2d):
                        maps = self.judge_group_ops(seq, maps)
                    elif isinstance(seq, avgpool_dropout_fc):
                        maps["AdaptiveAvgPool2d"] += 1
                        maps["Linear"] += 1
                        maps["Dropout"] += 1
                    else:
                        tmp = self.judge_single_op(seq, maps)
                        if tmp is None:
                            print("unimplement seq ", type(seq))
                        else:
                            maps = tmp
            elif isinstance(layer, Inception) or isinstance(layer, InceptionAux) or isinstance(layer, BasicConv2d):
                maps = self.judge_group_ops(layer, maps)
            else:
                tmp = self.judge_single_op(layer, maps)
                if tmp is None:
                    print("unimplement layer ", type(layer))
                else:
                    maps = tmp
            self.layer2op.append(maps)

    def check_type(self, m) -> bool:
        return isinstance(m, Inception)
        # return isinstance(m, Inception) or isinstance(m, InceptionAux)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * \
                (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * \
                (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * \
                (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Tensor, aux1: Optional[Tensor]) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> GoogLeNetOutputs:
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn(
                    "Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

    def get_submodules(self):
        return self.moduleList


class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(
                ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def googlenet(pretrained=False, progress=False, **kwargs) -> GoogLeNet:
    """GoogLeNet (Inception v1) model architecture from
    `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.GoogLeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.GoogLeNet_Weights
        :members:
    """

    model = GoogLeNet(**kwargs)
    # if pretrained:
    file_name = model_urls["googlenet"].split("/")[-1]
    model_path = os.path.join(model_dir, file_name)
    # print(model_urls[arch])
    state_dict = None
    if os.path.exists(model_path):
        # state_dict = load_state_dict_from_url(model_path)
        state_dict = torch.load(model_path)
    else:
        state_dict = load_state_dict_from_url(
            model_urls["googlenet"], progress=progress, check_hash=True, model_dir=model_dir)
    model.load_state_dict(state_dict)
    return model


# %%
if __name__ == '__main__':
    model_func = googlenet
    model = model_func().half().cuda().eval()
    _submodules = model.get_submodules()
    start = 10
    end = 17

    input = torch.rand(16, 3, 224, 224).half().cuda()
    # print(model(input))
    submodel1 = nn.Sequential(*_submodules[:start])
    inter_input = submodel1(input)
    submodel2 = nn.Sequential(*_submodules[start:])
    print(submodel2(inter_input))

    # after = nn.Sequential(*_submodules[start:end])
    # print(after(inter_input))

# %%
