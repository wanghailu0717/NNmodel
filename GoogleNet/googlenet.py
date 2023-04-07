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
# 引入 onnx 相关工具
import onnx
from onnxsim import simplify
from onnx import version_converter
from onnx2torch import convert
# 引入参数解析包
import argparse
# 引入工具包
import sys
from tqdm import tqdm

# 定义参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--train_batch", default="4", type=int, help="Default value of train_batch is 4.")
parser.add_argument("--train_epoch", default="2", type=int, help="Default value of train_epoch is 2. Set 0 to this argument if you want an untrained network.")
parser.add_argument("--infer_batch", default="1", type=int, help="Default value of infer_batch is 1.")
parser.add_argument("--which_device", type=str, default="cpu", choices=["mlu", "cuda", "cpu"])
parser.add_argument("--pytorch_infer", type=str, default="False", choices=["False", "True"])
parser.add_argument("--gofusion_infer", type=str, default="False", choices=["False", "True"])
parser.add_argument("--onnx_file", type=str, default="")
args = parser.parse_args()

print(vars(args))

# 定义网络并显示
class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x
net = GoogleNet(10)
print(net)

# 定义数据预处理方式以及训练集与测试集并进行下载
transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])
trainset = torchvision.datasets.CIFAR10(
                    root='../data', 
                    train=True, 
                    download=True,
                    transform=transform)
trainloader = torch.utils.data.DataLoader(
                    trainset, 
                    batch_size=args.train_batch,
                    shuffle=True, 
                    num_workers=2)
testset = torchvision.datasets.CIFAR10(
                    '../data',
                    train=False, 
                    download=True, 
                    transform=transform)
testloader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=args.infer_batch, 
                    shuffle=False,
                    num_workers=2)

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 定义硬件设备
device = torch.device(args.which_device)
input_rand = torch.zeros((1,3,32,32))
torch.onnx.export(net, input_rand, 'googlenet' + '_untrained' + '.onnx', input_names = ["image"], output_names = ["label"])
net = net.to(device)

# 网络训练
if args.train_epoch != 0:
    print("[INFO] Strat training " + "googlenet" + " network on " + args.which_device)
    start = time.time()
    for epoch in range(args.train_epoch):  
        print("[INFO] Training " + str(epoch) + " epoch...")
        running_loss = 0.0
        start_0 = time.time()
        for i, data in tqdm(enumerate(trainloader, 0)):
            # 输入数据
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播、计算损失、反向计算、参数更新
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 打印日志
            running_loss += loss.item()
            if i % 2000 == 1999: # 每2000个batch打印一下训练状态
                end_2000 = time.time()
                print('[%d, %5d] loss: %.3f take %.5f s' \
                      % (epoch+1, i+1, running_loss / 2000, (end_2000-start_0)))
                running_loss = 0.0
                start_0 = time.time()
    end = time.time()
    print('Finished Training: ' + str(end- start) + 's')
    input_rand = torch.zeros((1,3,32,32))
    net = net.to("cpu")
    torch.onnx.export(net, input_rand, 'googlenet' + '.onnx', input_names = ["image"], output_names = ["label"])

# # 网络推理
# correct = 0 # 预测正确的图片数
# total = 0 # 总共的图片数
# # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum()
# 
# print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))

###################################################################################
# Gofusion 运行
if args.gofusion_infer == "True":
    if len(args.onnx_file) != 0:
        model = onnx.load(args.onnx_file)
    else:
        model = onnx.load('./'+ 'googlenet' +'.onnx')
    model, check = simplify(model)

    from pyinfinitensor.onnx import OnnxStub, backend
    gofusion_model = OnnxStub(model, backend.bang_runtime())
    model = gofusion_model
    print("[INFO] Gofusion strat infer " + 'googlenet' + " network on " + args.which_device)
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    total_time = 0.0
    # 使用本项目的 Runtime 运行刚才加载并转换的模型, 验证是否一致
    for data in tqdm(testloader):
        images, labels = data
        next(model.inputs.items().__iter__())[1].copyin_float(images.reshape(-1).tolist())
        start_time = time.time()
        model.init()
        model.run()
        end_time = time.time()
        outputs = next(model.outputs.items().__iter__())[1].copyout_float()
        outputs = torch.tensor(outputs)
        outputs = torch.reshape(outputs,(1,10))
        total_time += (end_time - start_time)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('%d 张测试的准确率为: %f %%' % (total, 100 * correct / total))
    print('BatchSize = %d, GoFusion 推理耗时 %f s' % (labels.size(0), total_time / (total / (labels.size(0)))))
    del model, gofusion_model

###################################################################################
# Pytorch 运行
# 将模型转换为对应版本
if args.pytorch_infer == "True":
    if len(args.onnx_file) != 0:
        model = onnx.load(args.onnx_file)
    else:
        model = onnx.load('./'+ 'googlenet' +'.onnx')
    model, check = simplify(model)

    target_version = 13
    converted_model = version_converter.convert_version(model, target_version)
    torch_model = convert(converted_model)
    torch_model.to(device)
    torch_model.eval()
    print("[INFO] Pytorch strat infer " + "googlenet" + " network on " + args.which_device)
    correct = 0 # 预测正确的图片数
    total = 0 # 总共的图片数
    total_time = 0.0
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            start_time = time.time()
            outputs = torch_model(images)
            end_time = time.time()
            total_time += (end_time - start_time)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    print('%d 张测试的准确率为: %f %%' % (total, 100 * correct / total))
    print('BatchSize = %d, Pytorch 推理耗时 %f s' % (labels.size(0), total_time / (total / (labels.size(0)))))
    del torch_model, model

