# ref：https://blog.csdn.net/Peach_____/article/details/128758750
# %%
import torch
import torch.nn as nn
import random


def conv(in_channels, out_channels, kernel_size):  # conv+bn+leakyrelu
    padding = 1 if kernel_size == 3 else 0
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size, 1, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )


class Darknet19(nn.Module):  # darknet19
    def __init__(self):
        super(Darknet19, self).__init__()

        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = conv(3, 32, 3)
        self.conv2 = conv(32, 64, 3)
        self.bottleneck1 = nn.Sequential(
            conv(64, 128, 3),
            conv(128, 64, 1),
            conv(64, 128, 3)
        )
        self.bottleneck2 = nn.Sequential(
            conv(128, 256, 3),
            conv(256, 128, 1),
            conv(128, 256, 3)
        )
        self.bottleneck3 = nn.Sequential(
            conv(256, 512, 3),
            conv(512, 256, 1),
            conv(256, 512, 3),
            conv(512, 256, 1),
            conv(256, 512, 3)
        )
        self.bottleneck4 = nn.Sequential(
            conv(512, 1024, 3),
            conv(1024, 512, 1),
            conv(512, 1024, 3),
            conv(1024, 512, 1),
            conv(512, 1024, 3)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.bottleneck1(x)
        x = self.maxpool(x)
        x = self.bottleneck2(x)
        x = self.maxpool(x)
        shallow_x = self.bottleneck3(x)  # 浅层特征
        deep_x = self.maxpool(shallow_x)
        deep_x = self.bottleneck4(deep_x)  # 深层特征

        return shallow_x, deep_x


class Yolov2(nn.Module):
    def __init__(self):
        super(Yolov2, self).__init__()
        self.backbone = Darknet19()
        self.deep_conv = nn.Sequential(
            conv(1024, 1024, 3),
            conv(1024, 1024, 3)
        )
        self.shallow_conv = conv(512, 64, 1)
        self.final_conv = nn.Sequential(
            conv(1280, 1024, 3),
            nn.Conv2d(1024, 125, 1, 1, 0)
        )

    def passthrough(self, x):  # passthrough layer
        return torch.cat([x[:, :, ::2, ::2], x[:, :, ::2, 1::2], x[:, :, 1::2, ::2], x[:, :, 1::2, 1::2]], dim=1)

    def forward(self, x):
        shallow_x, deep_x = self.backbone(x)  # (B,512,26,26)、(B,1024,13,13)
        # (B,512,26,26)-->(B,64,26,26)
        shallow_x = self.shallow_conv(shallow_x)
        shallow_x = self.passthrough(shallow_x)  # (B,64,26,26)-->(B,256,13,13)
        deep_x = self.deep_conv(deep_x)  # (B,1024,13,13)-->(B,1024,13,13)
        # (B,1024,13,13)cat(B,256,13,13)-->(B,1280,13,13)
        feature = torch.cat([deep_x, shallow_x], dim=1)
        # (B,1280,13,13)-->(B,1024,13,13)-->(B,125,13,13)
        return self.final_conv(feature)


# %%
if __name__ == "__main__":
    batch_size = 8
    image_channels = 3
    image_size = random.randrange(320, 608 + 32, 32)  # [320,352,...,608]
    images = torch.randn(batch_size, image_channels, image_size, image_size)
    print(images.shape)
    yolov2 = Yolov2()
    print(yolov2(images).shape)

# %%
