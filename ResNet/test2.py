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
import cv2 as cv
from onnx import version_converter
from onnx2torch import convert

device= torch.device("cpu")

model = onnx.load('resnet18-v2-7.onnx')

# 将模型转换为对应版本
target_version = 13
converted_model = version_converter.convert_version(model, target_version)
torch_model = convert(converted_model)
torch_model.to(device)

img = cv.imread('2.jpg')
img = cv.resize(img, (224,224))
img = torch.as_tensor(img)
img = torch.reshape(img,((1,3,224,224)))
outputs = torch_model(img)
print(outputs)
_, predicted = torch.max(outputs, 1)


# 运行刚才加载并转换的模型, 验证是否一致

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))

