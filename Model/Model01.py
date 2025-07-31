import torch.nn as nn
import torchvision.datasets

from torch.nn import Sequential, MaxPool2d, Conv2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset,1)

class Model01(nn.Module):
    def __init__(self):
        super(Model01,self).__init__()
        self.model=Sequential(
            Conv2d(3,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,64,5,1,2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        return self.model(x)


if __name__ == '__main__':
    model01 = Model01()

    # loss = nn.L1Loss()
    # result = loss(outputs, targets)
    # mean abstract error(MAE)相减绝对值再平均,里面必须是浮点数

    # loss = nn.MSELoss()
    # result = loss(outputs, targets)
    # mean squared error(MSE)平均方差

    loss = nn.CrossEntropyLoss()

    for data in dataloader:
        images, targets = data
        outputs = model01(images)
        result = loss(outputs, targets)
        # 反向传播
        result.backward()
        # 优化器
        print(result)



        # writer=SummaryWriter("logs")
        # writer.add_graph(model01,input)
        # writer.close()

        # torch.onnx.export(model01, images, f"model01.onnx")
        # https://netron.app/