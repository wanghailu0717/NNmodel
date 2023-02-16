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
                    batch_size=4,
                    shuffle=True, 
                    num_workers=2)
testset = torchvision.datasets.CIFAR10(
                    '../data',
                    train=False, 
                    download=True, 
                    transform=transform)
testloader = torch.utils.data.DataLoader(
                    testset,
                    batch_size=4, 
                    shuffle=False,
                    num_workers=2)

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 定义硬件设备
device = torch.device("cpu")
net = net.to(device)

# 网络训练
start = time.time()
for epoch in range(5):  
    running_loss = 0.0
    start_0 = time.time()
    for i, data in enumerate(trainloader, 0):
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

# 网络推理
correct = 0 # 预测正确的图片数
total = 0 # 总共的图片数
# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
