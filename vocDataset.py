import os

import torch
import xmltodict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VocDataset(Dataset):
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
        self.classes_list = ["no helmet", "motor", "number", "with helmet"]
        # 所有标注类别



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
        label_name = image_name.split(".")[0]+".xml"
        # xml格式对应VOC格式标注文件
        label_path = os.path.join(self.label_folder,label_name)
        with open(label_path,"r",encoding="utf-8") as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)
        objects = label_dict["annotation"]["object"]
        # 获取到所有的标注
        target = []
        for object in objects:
            object_name = object["name"]
            object_class_id = self.classes_list.index(object_name)
            # 利用索引完成转换
            object_xmax = float(object["bndbox"]["xmax"])
            object_ymax = float(object["bndbox"]["ymax"])
            object_xmin = float(object["bndbox"]["xmin"])
            object_ymin = float(object["bndbox"]["ymin"])
            target.extend([object_class_id, object_xmax, object_ymax, object_xmin, object_ymin])
        target = torch.tensor(target)
        if self.image_transform:
            image = self.image_transform(image)
        return image, target




# 单元测试(输入main即可)
if __name__ == '__main__':
    train_dataset = VocDataset(r"E:\py project\torchProject\dataset\HelmetDataset-VOC\train\images",
                              r"E:\py project\torchProject\dataset\HelmetDataset-VOC\train\labels",
                               transforms.Compose([transforms.ToTensor()]),
                               None
                               )
    print(len(train_dataset))
    print(train_dataset[11])



