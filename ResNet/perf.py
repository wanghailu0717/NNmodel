# 引入 torch 包、神经网络包、函数包、优化器
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

# 引入图像包、图像处理包、显示包、时间包
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import time

# 引入onnx 相关工具
import onnx
from onnxsim import simplify
from onnx import version_converter
from onnx2torch import convert

# 定义数据预处理方式以及训练集与测试集并进行下载
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # 需要 resize 成当前模型使用的输入图片 shape 规格
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
    ]
)
trainset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=2
)
testset = torchvision.datasets.CIFAR10(
    "../data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2  # batchsize 需要根据模型 input 形状指定
)


model = onnx.load("./resnet18.onnx")

# 本项目对模型进行优化
model, check = simplify(model)
from pyinfinitensor.onnx import OnnxStub, backend

gofusion_model = OnnxStub(model, backend.cuda_runtime())
model = gofusion_model
model.init()

correct = 0  # 预测正确的图片数
total = 0  # 总共的图片数
total_time = 0.0
# 使用本项目的 Runtime 运行刚才加载并转换的模型, 验证是否一致
for data in testloader:
    images, labels = data
    next(model.inputs.items().__iter__())[1].copyin_float(images.reshape(-1).tolist())
    start_time = time.time()
    model.run()
    end_time = time.time()
    outputs = next(model.outputs.items().__iter__())[1].copyout_float()
    outputs = torch.tensor(outputs)
    outputs = torch.reshape(outputs, (4, 10))  # 根据 batchsize 进行 reshape 操作
    total_time += end_time - start_time
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print("10000张测试集中的准确率为: %f %%" % (100 * correct / total))
print(
    "BatchSize = %d, GoFusion 推理耗时 %f s"
    % (labels.size(0), total_time / (total / (labels.size(0))))
)
