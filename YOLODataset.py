import os

import torch
import xmltodict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class YOLODataset(Dataset):
    def __init__(self,image_folder,label_folder,image_transform,label_transform):
        # image_folder文件路径
        # abel_folder标注路径
        # image_transform文件图片变换
        # label_transform标注变换
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.image_names = os.listdir(self.image_folder)
        # self.image_name用于获取文件名称，以字典形式存放，以计算数据集长度




    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index):
        # 获取train_dataset[index]指向的文件
        image_name = self.image_names[index]
        # 得到self.image_names[index]这个文件的名称，下一步进行字符串拼接来找到文件
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")
        # 所有图片都转化为三通道RGB图片
        # 找图片对应的标注文件
        label_name = image_name.split(".")[0]+".txt"
        # xml格式对应VOC格式标注文件
        label_path = os.path.join(self.label_folder,label_name)
        with open(label_path,"r",encoding="utf-8") as f:
            label_content = f.read()
        object_infos = label_content.strip().split("\n")
        # 去除空字符串并分割
        target = []
        for object_info in object_infos:
            info_list = object_info.split(" ")
            class_id = float(info_list[0])
            center_x = float(info_list[1])
            center_y = float(info_list[2])
            width = float(info_list[3])
            height = float(info_list[4])
            target.extend([class_id, center_x, center_y, width, height])
        target = torch.tensor(target)
        if self.image_transform:
            image = self.image_transform(image)
        return image, target




# 单元测试(输入main即可)
if __name__ == '__main__':
    train_dataset = YOLODataset(r"E:\py project\torchProject\dataset\HelmetDataset-YOLO-Train\images",
                              r"E:\py project\torchProject\dataset\HelmetDataset-YOLO-Train\labels",
                                transforms.Compose([transforms.ToTensor()]),
                                None
                                )
    print(len(train_dataset))
    print(train_dataset[11])



