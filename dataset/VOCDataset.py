import os

import torch
import xmltodict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VocDataset(Dataset):
    def __init__(self, image_folder, label_folder, image_transform, label_transform):
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
        label_name = image_name.split(".")[0] + ".xml"
        # xml格式对应VOC格式标注文件
        label_path = os.path.join(self.label_folder, label_name)
        with open(label_path, "r", encoding="utf-8") as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)

        target = []
        # --- More Robust Parsing Logic ---
        # Safely get the 'annotation' dictionary
        annotation = label_dict.get("annotation")
        if annotation:
            # Safely get the 'object' list/dictionary
            objects = annotation.get("object")
            if objects:
                # *** Key Fix: Ensure objects is always a list ***
                if not isinstance(objects, list):
                    objects = [objects]  # If there's only one object, wrap it in a list

                for object_info in objects:
                    try:
                        object_name = object_info["name"]
                        object_class_id = self.classes_list.index(object_name)
                        bndbox = object_info["bndbox"]
                        object_xmax = float(bndbox["xmax"])
                        object_ymax = float(bndbox["ymax"])
                        object_xmin = float(bndbox["xmin"])
                        object_ymin = float(bndbox["ymin"])
                        target.extend([object_class_id, object_xmax, object_ymax, object_xmin, object_ymin])
                    except (KeyError, ValueError) as e:
                        print(f"\n[Warning] Skipping malformed object in '{label_name}'. Reason: {e}")
                        continue

        target = torch.tensor(target)
        if self.image_transform:
            image = self.image_transform(image)
        return image, target


# 单元测试(输入main即可)
if __name__ == '__main__':
    train_dataset = VocDataset(r"E:\py project\torchProject\data\HelmetDataset-VOC\train\images",
                               r"E:\py project\torchProject\data\HelmetDataset-VOC\train\labels",
                               transforms.Compose([transforms.ToTensor()]),
                               None
                               )
    print(len(train_dataset))
    # 测试一个已知只有一个物体的样本
    image, target = train_dataset[0]
    print("图片 0 的 Target:", target)
    assert target.numel() > 0, "单个物体的 target 不应为空！"

    # 测试一个已知有多个物体的样本
    image, target = train_dataset[11]
    print("图片 11 的 Target:", target)
    print("测试通过！")

