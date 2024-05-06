from resnet_splited import resnet18, resnet34, resnet50, resnet101, resnet152
from inception_splited import inception_v3
from vgg_splited import vgg16, vgg19
from yolo_splited import YoloBody
import torch
import time


if __name__ == '__main__':
    model = resnet18(False, True).cuda().eval()
    # model = vgg16(False, True).cuda().eval()
    # model = inception_v3(False, True, init_weights=True).cuda().eval()
    batch = 1
    for times in range(6):
        total = 0
        times = 0
        for i in range(1, 16):
            start = time.time()
            # input = torch.rand(batch, 3, 299, 299).cuda()
            input = torch.rand(batch, 3, 224, 224).cuda()
            output = model(input)
            end = time.time()
            # print("{} ms".format(1000*(end-start)))
            if i > 5:  # 6~15
                times += 1
                total += 1000*(end-start)
        print("batch:{} avg {} ms".format(batch, total/times))
        batch *= 2

# resnet152+inception_v3

# batch1
# 21281.591836734693 vs. 13311.908163265307vs. 21189.877551020407
# resnet152,0,52,1,0,inception_v3,0,0,1,0,21372.5,21281.591836734693,362.924932833412
# resnet152,0,0,1,0,inception_v3,0,14,1,0,13291.0,13311.908163265307,307.4150649826047
# resnet152,0,52,1,0,inception_v3,0,14,1,0,21050.5,21189.877551020407,877.6746789950731


# batch2
# 20346.34693877551 vs. 13259.938775510203 vs. 20644.98979591837
# resnet152,0,52,2,0,inception_v3,0,0,2,0,20118.0,20346.34693877551,1055.4109955033653
# resnet152,0,0,2,0,inception_v3,0,14,2,0,13247.0,13259.938775510203,154.75430272126692
# resnet152,0,52,2,0,inception_v3,0,14,2,0,20536.5,20644.98979591837,694.1888600123632


# batch4
# 19586.775510204083 vs. 12716.387755102041 vs. 19966.030612244896
# resnet152,0,52,4,0,inception_v3,0,0,4,0,19599.0,19586.775510204083,284.7253789901178
# resnet152,0,0,4,0,inception_v3,0,14,4,0,12738.5,12716.387755102041,242.8603206789173
# resnet152,0,52,4,0,inception_v3,0,14,4,0,19272.5,19966.030612244896,1947.9506837725205


# batch8
# 17870.20408163265 vs. 12340.785714285714 vs. 24313.877551020407
# resnet152,0,52,8,0,inception_v3,0,0,8,0,17717.0,17870.20408163265,747.6136588108079
# resnet152,0,0,8,0,inception_v3,0,14,8,0,12208.5,12340.785714285714,979.2558529364039
# resnet152,0,52,8,0,inception_v3,0,14,8,0,23850.0,24313.877551020407,1955.6544667117228


# batch16
# 23084.336734693876 vs. 14433.030612244898 vs. 38179.91836734694
# resnet152,0,52,16,0,inception_v3,0,0,16,0,22458.0,23084.336734693876,1749.1184423546783
# resnet152,0,0,16,0,inception_v3,0,14,16,0,13824.0,14433.030612244898,1517.1202275288456
# resnet152,0,52,16,0,inception_v3,0,14,16,0,37851.5,38179.91836734694,1927.1111544224361

# batch32
# 43339.80612244898 vs. 26089.01020408163 vs. 72505.13265306123
# resnet152,0,52,32,0,inception_v3,0,0,32,0,42985.5,43339.80612244898,1740.3261051330046
# resnet152,0,0,32,0,inception_v3,0,14,32,0,25542.0,26089.01020408163,1859.4953588787994
# resnet152,0,52,32,0,inception_v3,0,14,32,0,72238.5,72505.13265306123,1730.837968800055

# if __name__ == '__main__':
#     model = YoloBody(False, True).cuda().eval()
#     # model = vgg16(False, True).cuda().eval()
#     # model = inception_v3(False, True, init_weights=True).cuda().eval()
#     total = 0
#     for i in range(1, 16):
#         start = time.time()
#         # input = torch.rand(8, 3, 299, 299).cuda()
#         input = torch.rand(8, 3, 224, 224).cuda()
#         output = model(input)
#         end = time.time()
#         print("{} ms".format(1000*(end-start)))
#         if i > 5:  # 6~15
#             total += 1000*(end-start)
#     print("avg {} ms".format(total/10))
