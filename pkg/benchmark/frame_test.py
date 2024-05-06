
# %%
import onnxruntime as ort
import numpy as np
import time
import torch
# 将torch的预训练模型转为onnx格式
import torch
from torchvision.models import resnet50
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
pytorch_model_path = '/home/onceas/wanna/mt-dnn/model/plain/resnet50-0676ba61.pth'


def parse_onnx_path(onnx_name):
    return os.path.join(os.path.dirname(__file__),
                        "../../model/onnx/", onnx_name+".onnx")


def export_onnx():
    state_dict = torch.load(pytorch_model_path)
    model = resnet50()
    model.load_state_dict(state_dict)
    model.cuda().eval()
    data = torch.rand(batch_size, 3, 224, 224).cuda()
    # torch.save(model, 'resnet.pth')
    torch.onnx.export(model, data, parse_onnx_path("resnet50"), opset_version=7,
                      input_names=['input'], output_names=['output'])


def test_torch():
    state_dict = torch.load(pytorch_model_path)
    model = resnet50()
    model.load_state_dict(state_dict)
    model.cuda().eval()
    for i in range(2):
        input = torch.rand(batch_size, 3, 224, 224).cuda()
        res = model(input)
    exec_times = np.array([])
    for i in range(10):
        init_input = torch.rand(batch_size, 3, 224, 224)
        start = time.time()
        input = init_input.cuda()
        res = model(input)
        # res = model.forward(input)
        exec_times = np.append(exec_times, time.time()-start)
    print(exec_times)
    print(1000*np.mean(exec_times))


def test_onnx():
    # Load the pre-trained ResNet50 model in ONNX format
    onnx_model_path = parse_onnx_path("resnet50")
    providers = ['CUDAExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers,sess_options=sess_options)  

    img = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    # Run inference using ONNX Runtime on GPU
    # providers = ['CPUExecutionProvider']
    input_name = ort_session.get_inputs()[0].name
    print(ort_session.get_inputs()[0].shape)
    out = ort_session.run(None, {input_name: img})[0]
    out = ort_session.run(None, {input_name: img})[0]
    times = []
    for i in range(10):
        img = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        start = time.time()
        out = ort_session.run(None, {input_name: img})[0]
        times.append(time.time()-start)
    print(times)
    print(1000*np.array(times).mean())


if __name__ == '__main__':
    export_onnx()
    test_onnx()
    test_torch()


# %%
