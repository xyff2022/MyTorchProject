import torch
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, LeakyReLU, Flatten, Linear, Dropout


class YOLOv1(Module):
    def __init__(self):
        super().__init__()
        self.YOLO = Sequential(
            Conv2d(3, 64, 7, 2, 3),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),
            Conv2d(64, 192, 3, 1, 1),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),
            Conv2d(192, 128, 1),
            LeakyReLU(0.1),
            Conv2d(128, 256, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(256, 256, 1),
            LeakyReLU(0.1),
            Conv2d(256, 512, 3, 1, 1),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),
            Conv2d(512, 256, 1, 1),
            LeakyReLU(0.1),
            Conv2d(256, 512, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 256, 1, 1),
            LeakyReLU(0.1),
            Conv2d(256, 512, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 256, 1, 1),
            LeakyReLU(0.1),
            Conv2d(256, 512, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 256, 1, 1),
            LeakyReLU(0.1),
            Conv2d(256, 512, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 512, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),
            Conv2d(1024, 512, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(1024, 512, 1, 1),
            LeakyReLU(0.1),
            Conv2d(512, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, 3, 2, 1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, 3, 1, 1),
            LeakyReLU(0.1),
            Flatten(),
            Linear(50176, 4096),
            LeakyReLU(0.1),
            Dropout(0.5),
            Linear(4096, 1470),
        )

    def forward(self, x):
        return self.YOLO(x)

if __name__ == '__main__':
    yolo = YOLOv1()
    dummy_input = torch.randn(4, 3, 448, 448)
    out = yolo(dummy_input)
    out = out.view(-1, 7, 7, 30)
    print(out.shape)