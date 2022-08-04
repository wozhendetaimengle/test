import torch
import time
import os
from PIL import Image
import  cv2
from torchvision import transforms
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
device = torch.device("cuda:1")
# create some regular pytorch model...
model = alexnet(pretrained=True).eval().cuda()
# create example data

img_path='./1656552746_7f8680cef81411ecb9c400163e16ff5d.jpg'

img = cv2.imread(img_path)
img = cv2.resize(img, (244,244))
img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)/255
print(img.size)
x = img.to(device)

x = torch.ones((1, 3, 224, 224)).cuda()

model_trt = TRTModule()
# convert to TensorRT feeding sample data as input
model_trt.load_state_dict(torch.load('alexnet_trt.pth'))


torch.cuda.synchronize()
start = time.time()
result = model(x)
torch.cuda.synchronize()
end = time.time()
print(end - start)


torch.cuda.synchronize()
start = time.time()
y = model_trt(x)
torch.cuda.synchronize()
end = time.time()
print(end - start)

