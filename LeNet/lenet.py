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
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(16*5*5, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x): 
        x = func.max_pool2d(func.relu(self.conv1(x)), (2, 2)) 
        x = func.max_pool2d(func.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1) 
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)        
        return x
net = Net()
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
torch.onnx.export(net, input_rand, 'lenet' + '_untrained' + '.onnx', input_names = ["image"], output_names = ["label"])
net = net.to(device)

# 网络训练
if args.train_epoch != 0:
    print("[INFO] Strat training " + "lenet" + " network on " + args.which_device)
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
    torch.onnx.export(net, input_rand, 'lenet' + '.onnx', input_names = ["image"], output_names = ["label"])

# # 网络推理
# correct = 0 # 预测正确的图片数
# total = 0 # 总共的图片数
# # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
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
        model = onnx.load('./'+ 'lenet' +'.onnx')
    model, check = simplify(model)

    from pyinfinitensor.onnx import OnnxStub, backend
    gofusion_model = OnnxStub(model, backend.bang_runtime())
    model = gofusion_model
    print("[INFO] Gofusion strat infer " + 'lenet' + " network on " + args.which_device)
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
        model = onnx.load('./'+ 'lenet' +'.onnx')
    model, check = simplify(model)

    target_version = 13
    converted_model = version_converter.convert_version(model, target_version)
    torch_model = convert(converted_model)
    torch_model.to(device)
    torch_model.eval()
    print("[INFO] Pytorch strat infer " + "lenet" + " network on " + args.which_device)
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

