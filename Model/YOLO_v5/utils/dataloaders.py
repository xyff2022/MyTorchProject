import os
import random
from multiprocessing.pool import ThreadPool

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# (我们可以在这里定义 PIN_MEMORY)
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'

# (新增) 定义线程和分布式训练相关的常量
NUM_THREADS = min(6, max(1, os.cpu_count() - 1))  # 线程数
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))     # DDP 进程 rank, -1 表示非 DDP
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'   # tqdm 进度条格式

def create_dataloader(path,
                      img_size,
                      batch_size,
                      stride,
                      augment=False,
                      cache=False,
                      pad=0.0,
                      workers=8,
                      shuffle=False):
    """
        (高级经理) - 创建并封装 LoadImagesAndLabels
        """

    # ==========================================================
    # 4.1 实例化数据集 (厨房)
    # ==========================================================

    # (在 DDP 模式下，这里会有一个 'with torch_distributed_zero_first(rank):')
    # (我们直接创建它)
    dataset = LoadImagesAndLabels(
        path,
        img_size,
        batch_size,
        augment=augment,
        cache_images_in_ram=True,
        stride=stride,
        pad=pad)

    # ==========================================================
    # 4.2 计算工作进程数 (nw)
    # ==========================================================

    # (安全检查：取 CPU核心数、批量大小 和 用户指定 的最小值)
    # (这可以防止 'num_workers' 设置过高)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    batch_size = min(batch_size, len(dataset))
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205)

    # ==========================================================
    # 4.3 创建 DataLoader (流水线)
    # ==========================================================

    # (这是 PyTorch 的核心)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=nw,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn,
                        worker_init_fn=seed_worker,
                        generator=generator
                        )

    # ==========================================================
    # 4.4 返回
    # ==========================================================

    return loader, dataset


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =============================== 主类: YOLOv3Dataset ===============================

class LoadImagesAndLabels(Dataset):
    """
    YOLOv5 官方数据集类 (LoadImagesAndLabels)
    负责扫描文件、缓存标签、并在 __getitem__ 中加载和处理单张图片。
    """

    def __init__(self,
                 img_root,
                 img_size=640,
                 batch_size=16,
                 augment=False,
                 cache_images_in_ram=False,
                 stride=32,
                 pad=0.0,
                 mosaic_random = 1.0,
                 mixup_random = 0.1):

        # ==========================================================
        # 1.1 初始化 - 存储配置
        # ==========================================================

        # 存储传入的参数
        self.img_root = img_root
        self.img_size = img_size
        self.augment = augment
        self.stride = stride
        self.image_files = []  # 存储所有图片路径
        self.label_files = []  # 存储所有标签路径
        self.mosaic_random = mosaic_random
        self.mixup_random = mixup_random
        self.mosaic = augment

        # ==========================================================
        # 1.2 扫描文件路径
        # ==========================================================

        # 自动扫描图片文件夹, 获取所有图片文件名
        self.image_files = sorted([name for name in os.listdir(img_root)
                                   if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

        self.n = len(self.image_files)
        # ==========================================================
        # 1.3 查找对应的标签路径
        # ==========================================================

        # (这是一个巧妙的技巧：将 'images' 替换为 'labels', 'jpg' 替换为 'txt')
        label_dir = os.path.join(os.path.dirname(img_root), 'labels')  # 从 '.../images' 变为 '.../labels'
        self.label_files = [
            os.path.join(label_dir, x.replace(os.path.splitext(x)[-1], '.txt'))
            for x in self.image_files
        ]



        # ==========================================================
        # 1.4 缓存标签 (关键步骤) (已修改: 同时缓存图片尺寸)
        # ==========================================================

        # 图片缓存逻辑 (RAM 缓存)
        # 如果 cache_images_in_ram=True, 则初始化一个列表来存储缓存的图片
        self.images = [None] * self.n if cache_images_in_ram else None  # (L1: 原始图片RAM缓存)
        self.im_hw0 = [None] * self.n if cache_images_in_ram else None  # (L1: 原始尺寸缓存)
        self.im_hw = [None] * self.n if cache_images_in_ram else None  # (L1: 缩放后尺寸缓存)

        # 定义缓存文件的保存路径
        cache_path = os.path.join(os.path.dirname(img_root), 'data.cache')

        if os.path.exists(cache_path):
            print(f'Loading data from cache: {cache_path}')
            cache = np.load(cache_path, allow_pickle=True).item()
            self.labels = cache['labels']
            self.shapes = cache['shapes']
        else:
            # cache_labels_and_shapes_in_disk 现在返回两个字典,使用image_path索引
            self.labels, self.shapes = self.cache_labels_and_shapes_in_disk(cache_path)




        if cache_images_in_ram:
            # 开启线程池
            print(f'Caching images in RAM using {NUM_THREADS} threads...')
            results = ThreadPool(NUM_THREADS).imap(self.load_image, range(self.n))
            pbar = tqdm(enumerate(results),
                        total=self.n,
                        bar_format=TQDM_BAR_FORMAT,
                        disable=LOCAL_RANK > 0)  # (只在主进程显示进度条)

            # 收集结果
            for i, result in pbar:
                self.images[i], self.im_hw0[i], self.im_hw[i] = result

            pbar.close()
            print("Images cached successfully.")

        self.mosaic_transform = A.Compose([
            # --- 几何变换 (在 1280x1280 上) ---
            A.Affine(
                scale=(0.7, 1.5),# 随机缩放: 70% 到 150%
                translate_percent=(-0.05, 0.05), # 随机平移: -10% 到 +10%
                rotate=(0.0, 0.0),  # 随机旋转
                shear=(0.0, 0.0),  # 随机错切
                p=1.0 # 有 100% 的概率应用 Affine 变换
            ),

            # --- 透视变换  ---
            A.Perspective(p=0.0),  # 以p的概率实现

            # --- 裁剪 (复现 border=[-320, -320]) ---
            # (从 1280x1280 裁剪中心 640x640)
            A.CenterCrop(height=img_size, # 目标高度 (e.g., 640)
                         width=img_size, # 目标宽度 (e.g., 640)
                         p=1.0 # 100% 总是执行
            ),

            # --- 颜色变换 (在 640x640 上) ---
            # A.ColorJitter 集合了 亮度、对比度、饱和度、色调 变换
            A.ColorJitter(
                brightness=0.4, # 亮度
                contrast=0.4,  # 对比度
                saturation=0.7, # 饱和度
                hue=0.015,  # 色调
                p=0.75  # 有 75% 的概率应用 ColorJitter
            ),

            # 2. 额外的亮度和对比度 (YOLOX/v8 风格)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # ( -20% to +20% )
                contrast_limit=0.2,  # ( -20% to +20% )
                p=0.5  # 50% 概率应用
            ),

            # --- 随机模糊  ---
            A.Blur(blur_limit=(3, 7),# 模糊核的大小在 3x3 到 7x7 之间
                    p=0.1    # 有 10% 的概率应用模糊
            ),

            # 4. 随机转灰度
            A.ToGray(
                p=0.01  # 1% 概率应用
            ),

            # --- 翻转 (在 640x640 上) ---
            A.HorizontalFlip(p=0.5),  # 50% 概率水平翻转
            A.VerticalFlip(p=0.0),  # 0% 概率垂直翻转

            # --- 格式化 (内联) ---
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)

        ], bbox_params=A.BboxParams(format='pascal_voc',  # 我们输入的是 xyxy
                                    label_fields=['classes'],
                                    min_visibility=0.1,  # 裁剪后可见度低于 10% 的框将被移除
                                    min_area=10))  # 面积小于 10 像素的框将被移除

        self.single_image_transform = A.Compose([
            # --- 几何变换  ---
            A.Affine(
                # scale=(0.7, 1.5),  # 随机缩放: 70% 到 150%
                border_mode=cv2.BORDER_CONSTANT,
                fill=(114, 114, 114),
                translate_percent=(-0.1, 0.1),  # 随机平移: -10% 到 +10%
                rotate=(0.0, 0.0),  # 随机旋转
                shear=(0.0, 0.0),  # 随机错切
                p=1.0 , # 有 100% 的概率应用 Affine 变换
            ),

            # --- 透视变换  ---
            A.Perspective(p=0.0),  # 以p的概率实现

            # --- 颜色变换 (在 640x640 上) ---
            # A.ColorJitter 集合了 亮度、对比度、饱和度、色调 变换
            A.ColorJitter(
                brightness=0.4,  # 亮度
                contrast=0.4,  # 对比度
                saturation=0.7,  # 饱和度
                hue=0.015,  # 色调
                p=0.75  # 有 75% 的概率应用 ColorJitter
            ),

            # 2. 额外的亮度和对比度 (YOLOX/v8 风格)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # ( -20% to +20% )
                contrast_limit=0.2,  # ( -20% to +20% )
                p=0.5  # 50% 概率应用
            ),

            # --- 随机模糊  ---
            A.Blur(blur_limit=(3, 7),  # 模糊核的大小在 3x3 到 7x7 之间
                   p=0.1  # 有 10% 的概率应用模糊
                   ),

            # 4. 随机转灰度
            A.ToGray(
                p=0.01  # 1% 概率应用
            ),

            # --- 翻转 (在 640x640 上) ---
            A.HorizontalFlip(p=0.5),  # 50% 概率水平翻转
            A.VerticalFlip(p=0.0),  # 0% 概率垂直翻转

            # --- 格式化  ---
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)

        ], bbox_params=A.BboxParams(  # [!!!] 边界框参数 [!!!]
            format='pascal_voc',
            label_fields=['classes'],
            min_visibility=0.1,
            min_area=10
        ))

        self.single_image_val_transform = A.Compose([
            # --- 格式化 (内联) ---
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)

        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['classes']
            # (验证时不过滤标签)
        ))


    def __len__(self):
        """返回数据集中图片的数量。"""
        return len(self.image_files)

    def __getitem__(self, index):

        # 随机应用mosaic和mixup
        mosaic = random.random() < self.mosaic_random and self.mosaic
        if mosaic:
            image, bboxes, class_labels = self.load_mosaic(index)
            shapes = None

            if random.random() < self.mixup_random:
                image1, bboxes1, class_labels1 = self.load_mosaic(random.randint(0, self.n - 1))
                image, bboxes, class_labels = mixup(image, bboxes, class_labels, image1, bboxes1, class_labels1)

            # 应用 Mosaic 变换管道
            # (输入 1280x1280, 输出 640x640 Tensor)
            try:
                transformed = self.mosaic_transform(image=image, bboxes=bboxes, classes=class_labels)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_classes = transformed['classes']
            except Exception as e:
                # 如果变换失败 (例如所有 bboxes 都被裁掉), 加载一个新样本
                print(f"警告: Mosaic 变换失败 ({e})。正在尝试下一个索引。")
                return self.__getitem__((index + 1) % self.n)
        else:
            # 加载单张图像 ---

            # 1. 加载原始图像
            # image 是 (h, w, 3), (h, w) 是原始尺寸
            image, (h0, w0), (h, w) = self.load_image(index)

            # 2. Letterbox (缩放和填充)
            # (scaleup=self.augment 意味着只在训练时放大图像)
            # image 变成了 (s, s, 3) (例如 640, 640, 3)
            # ratio 是缩放比例, (padw, padh) 是*总*填充量
            image, ratio, (dw, dh) = letterbox(image, (self.img_size, self.img_size))

            labels = self.labels[self.image_files[index]].copy()
            bboxes = np.zeros((0, 4))
            classes = np.zeros((0))

            if labels.size > 0:
                classes = labels[:, 0]
                bboxes_xywhn = labels[:, 1:]

                bboxes = xywhn2xyxy(bboxes_xywhn, w, h, dw / 2, dh / 2)

            # 1d. [!!! 核心 !!!] 应用单张图变换管道
            # (输入 640x640, 输出 640x640 Tensor)
            try:
                if self.augment:
                    transformed = self.single_image_transform(image=image, bboxes=bboxes, classes=classes)
                else:
                    transformed = self.single_image_val_transform(image=image, bboxes=bboxes, classes=classes)
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_classes = transformed['classes']
            except Exception as e:
                # 如果变换失败, 加载一个新样本
                print(f"警告: 单张图变换失败 ({e})。正在尝试下一个索引。")
                return self.__getitem__((index + 1) % self.n)

        # --- 3. 格式化标签 (转换为 YOLO 格式) ---

        nl = len(transformed_classes)  # 标签数量
        labels_out = torch.zeros((nl, 6))  # [img_idx, cls, xn, yn, wn, hn]

        if nl > 0:
            # 3a. 将 bboxes (xyxy 像素) 转换回 (xywhn 归一化)
            # h, w 此时 *必须* 是 img_size (e.g., 640, 640)
            h, w = self.img_size, self.img_size

            # (注意: transformed_bboxes 已经是 NumPy 数组或列表)
            bboxes_xyxy = np.array(transformed_bboxes)

            # 3b. 调用辅助函数 xyxy2xywhn
            bboxes_xywhn = xyxy2xywhn(bboxes_xyxy, w, h, clip=True)

            # 3c. 填充 labels_out
            labels_out[:, 1] = torch.tensor(transformed_classes)
            labels_out[:, 2:] = torch.from_numpy(bboxes_xywhn)

        # --- 4. 返回最终结果 ---
        # transformed_image 已经是 (C, H, W) 格式的 Tensor (来自 ToTensorV2)
        return transformed_image, labels_out



    def load_mosaic(self, index):
        """
        加载 4-mosaic 图像。
        返回 (img4, bboxes_xyxy, class_labels)
        """
        # bboxes4: 准备一个空列表，用于存储所有边界框 (xyxy 像素格式)
        bboxes4 = []
        # classes4: 准备一个空列表，用于存储所有类别 ID
        classes4 = []

        # s = 获取目标图像尺寸 (例如 640)，s 代表 'size'
        s = self.img_size

        # 1. 确定 Mosaic 拼接的中心点 (xc, yc)
        # s = 640, self.mosaic_border = [-320, -320]
        # random.uniform(320, 1280-320) => [320, 960]
        # 这使得4张图的拼接点在 (640, 640) 附近随机抖动
        xc = int(random.uniform(0.5 * s, 1.5 * s))
        yc = int(random.uniform(0.5 * s, 1.5 * s))

        # 2. 准备 4 张图像的索引
        # indices[0] = 当前 __getitem__ 请求的图像
        # indices[1,2,3] = 随机从数据集中选择的 3 张图像
        # self.indices 是一个包含所有图像索引的列表, e.g., [0, 1, ..., n-1]
        indices = [index] + random.choices(range(self.n), k=3)

        # 随机打乱这 4 张图的顺序，以决定它们各自被放置在哪个角落
        random.shuffle(indices)

        # 3. 遍历 4 张图像并进行拼接
        # i 会是 0, 1, 2, 3
        # index 是我们随机打乱后的图像索引
        for i, index in enumerate(indices):
            # 加载预处理（缩放）过的图像及其原始尺寸
            # img 是小图, (h, w) 是小图的尺寸 (e.g., 580, 640)
            image, _,  (h, w) = self.load_image(index)

            # 4. 分配 4 个位置 (左上, 右上, 左下, 右下)
            if i == 0:  # i == 0, 我们分配到左上角区域
                # 创建一个 2s x 2s x 3的灰色(114)大画布
                # (例如 1280 x 1280 x 3)
                image4 = np.full((2 * s, 2 * s, image.shape[2]), 114, dtype = np.uint8)

                # --- 计算坐标 ---
                # (xc, yc) 是我们在第2部分计算的随机中心点 (e.g., 500, 600)
                mosaic_xmax = xc
                mosaic_xmin = max(xc - w, 0)
                mosaic_ymax = yc
                mosaic_ymin = max(yc - h, 0)

                source_xmax = w
                source_xmin = max(w - xc, 0)
                source_ymax = h
                source_ymin = max(h - yc, 0)
            elif i == 1:
                mosaic_xmax = min(xc + w, 2 * s)
                mosaic_xmin = xc
                mosaic_ymax = yc
                mosaic_ymin = max(yc - h, 0)

                source_xmax = min(2 * s - xc, w)
                source_xmin = 0
                source_ymax = h
                source_ymin = max(h - yc, 0)

            elif i == 2:
                mosaic_xmax = xc
                mosaic_xmin = max(xc - w, 0)
                mosaic_ymax = min(yc + h, 2 * s)
                mosaic_ymin = yc

                source_xmax = w
                source_xmin = max(w - xc, 0)
                source_ymax = min(h, 2 * s - yc)
                source_ymin = 0

            elif i == 3:
                mosaic_xmax = min(xc + w, 2 * s)
                mosaic_xmin = xc
                mosaic_ymax = min(yc + h, 2 * s)
                mosaic_ymin = yc

                source_xmax = min(w, 2 * s - xc)
                source_xmin = 0
                source_ymax = min(h, 2 * s - yc)
                source_ymin = 0

            # img4[目标_y_min:目标_y_max, 目标_x_min:目标_x_max] = img[源_y_min:源_y_max, 源_x_min:源_x_max]
            image4[mosaic_ymin:mosaic_ymax, mosaic_xmin:mosaic_xmax] = image[source_ymin:source_ymax, source_xmin:source_xmax]

            # 计算标签的偏移量
            # 这是将 "小图坐标系" 转换到 "大画布坐标系" 的关键
            # padding_x = 目标 x_min - 源 x_min
            # padding_y = 目标 y_min - 源 y_min
            padding_x = mosaic_xmin - source_xmin
            padding_y = mosaic_ymin - source_ymin

            # 加载并转换标签坐标
            label = self.labels[self.image_files[index]].copy()
            if label.size:
                xywh = label[:, 1:]
                classes = label[:, 0]
                bboxes_xyxy = xywhn2xyxy(xywh, w, h, padding_x, padding_y)

                # 存储转换后的标签, 供后续合并
                bboxes4.append(bboxes_xyxy)
                classes4.append(classes)

        bboxes4 = np.concatenate(bboxes4, 0) if bboxes4 else np.zeros((0, 4))
        classes4 = np.concatenate(classes4, 0) if classes4 else np.zeros((0))

        # 裁剪超出大画布 (2s x 2s) 范围的边界框
        # 确保所有坐标都在 [0, 2*s] (例如 [0, 1280]) 的有效范围内
        np.clip(bboxes4, 0, 2 * s, out=bboxes4)  # 'out=bboxes4' 表示原地修改

        # 10. 返回结果
        # img4: (1280, 1280, 3) 的马赛克图像
        # bboxes4: (N, 4) 的像素坐标 (xyxy) 标签
        # classes4: (N,) 的类别 ID
        return image4, bboxes4, classes4



    def load_image(self, i):
        """"
        (由 __init__ 和 __getitem__ 调用)
        按 优先级 1(RAM) -> 2(data.cache) 加载图片
        返回 (im, hw_original, hw_resized)
        """
        # 优先级 1: 检查 RAM (self.ims)
        image = self.images[i]
        if image is not None:
            # 缓存命中! (最快)
            # (注意: __init__ 缓存时 im_hw0[i] 和 im_hw[i] 可能还是 None,
            #  但在第二次调用时(例如 __getitem__), 它们已经被填充了)
            if self.im_hw[i] is not None:
                return self.images[i], self.im_hw0[i], self.im_hw[i]
                # (如果im_hw0[i]是None, 说明是__init__第一次调用, 继续往下执行)

        # 优先级 2: 从原始 JPG 读取 (最慢)
        # 获取文件的名字，用来去硬盘中读取原始数据
        f = os.path.join(self.img_root, self.image_files[i])
        # f = self.image_files[i]

        image = cv2.imread(f) # BGR
        assert image is not None, f'Image Not Found {f}'
        (h0, w0) = image.shape[:2]  # 原始 hw

        # 执行"矩形缩放", 将最长边缩放到img_size
        r = self.img_size / max(h0, w0)
        if r != 1.0:
            interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
            # 在训练时，DataLoader 的速度是瓶颈，我们希望一切都尽可能快，所以用 INTER_LINEAR。
            # r > 1 (放大时)：当放大图片时，INTER_LINEAR 的效果很好，而且速度快。
            # 当缩小图片时，它的效果通常被认为比INTER_LINEAR更好，可以避免产生波纹（moirépatterns）。但它可能稍慢一点。
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation = interp)

        (h, w) = image.shape[:2]  # 缩放后的 hw

        return image, (h0, w0), (h, w)


    def cache_labels_and_shapes_in_disk(self, path):

        # 如果缓存不存在，则创建它
        print(f'Caching data to {path} (first time)...')
        # (修正) 初始化包含两个子字典的 cache
        cache = {'labels': {}, 'shapes': {}}

        # (tqdm 用于显示进度条)
        # (修正) enumerate 从 0 开始
        for i, (img_path, label_path) in enumerate(tqdm(zip(self.image_files, self.label_files), desc='Caching data', total=len(self.image_files))):
            try:
                # ------------------------------------
                # 1. 缓存标签
                # ------------------------------------
                with open(label_path) as f:
                    l = [x.split() for x in f.read().strip().splitlines()]
                    # 转换成 [N, 5] 的 NumPy 数组 [cls, x, y, w, h]
                    l = np.array(l, dtype=np.float32)

                if len(l) == 0:
                    l = np.empty((0, 5), dtype=np.float32)

                # (修正) 存入 'labels' 子字典
                cache['labels'][img_path] = l

                # ------------------------------------
                # 2. (新增) 缓存图片尺寸
                # ------------------------------------
                # (我们调用 load_image 来安全地获取尺寸)
                # (这会读取图像，但只在第一次缓存时发生)
                img, (h0, w0), _ = self.load_image(i)
                # (修正) 存入 'shapes' 子字典
                cache['shapes'][img_path] = (h0, w0)

            except Exception as e:
                print(f'ERROR: caching data for {img_path}: {e}')
                # (修正) 存入空值
                cache['labels'][img_path] = np.empty((0, 5), dtype=np.float32)
                cache['shapes'][img_path] = (0, 0)  # 存入无效尺寸

        # 保存缓存字典到 .npy 文件
        np.save(path, cache)
        print(f'Data cached successfully.')
        # (返回两个字典)
        return cache['labels'], cache['shapes']



    @staticmethod
    def collate_fn(batch):
        """
        自定义的批次打包函数, 用于处理YOLOv3的标签。
        """
        images, labels = zip(*batch)  # 解包一个批次的数据

        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                label[:, 0] = i

        # 过滤掉没有标签的样本, 并将所有标签拼接成一个大张量
        valid_labels = [label for label in labels if label.shape[0] > 0]

        if len(valid_labels) > 0:
            return torch.stack(images, 0), torch.cat(valid_labels, 0)
        else:
            return torch.stack(images, 0), torch.zeros((0, 6))




def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    将图像缩放并填充以适应新的形状，保持宽高比。
    这是YOLOv5 v7.0中 `utils/augmentations.py` 里的官方实现。

    输入参数:
        img (np.ndarray): 原始图像 (OpenCV格式, BGR, HWC)。
        new_shape (tuple): 你希望的目标尺寸，通常是一个元组 (height, width)，例如 (640, 640)
        color (tuple): 填充“黑边”时使用的 BGR 颜色值，默认为 (114, 114, 114)。
        auto (bool):
            - False (推理模式): 保证严格填充到 new_shape (例如 640x640)。
            - True (训练模式): 会最小化填充，一条边只填充到能被 stride 整除的最小尺寸（例如 640x384）。
        scaleFill (bool):
            - True:  不等比拉伸。直接将图像拉伸到 new_shape，忽略宽高比。
            - False: 等比缩放。保持宽高比。
        scaleup (bool):
            - True:  允许图像被放大（如果它小于 new_shape）。
            - False: 只缩小不放大。
        stride (int): 步长。`auto=True` 时，填充的尺寸会是 `stride` 的倍数。

    输出:
        img (np.ndarray): 经过 letterbox 处理后的图像。
        ratio (float): 原始尺寸与新尺寸的缩放比例 (new_shape / old_shape)。
        (dw, dh) (tuple): 左右 (dw) 和 上下 (dh) 填充的像素值总和。
    """

    # ==================================================================
    # 1. 获取当前形状并计算缩放比例 (r)
    # ==================================================================
    # 获取原始形状 [height, width]
    # 先进行缩放，然后填充,影响缩放scaleup
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # scaleup=False 意味着我们只缩小，不放大。
    # 如果 r > 1 (即图像在放大)，并且不允许 scaleup，我们就将 r 设为 1 (保持原样)
    if not scaleup:
        r = min(1.0, r)

    # ==================================================================
    # 2. 计算最终的填充量 (dw, dh)
    # ==================================================================


    # 计算缩放后、填充前的尺寸 (宽度, 高度) 这样做是为了计算填充量
    new_un_pad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    # dw, dh 是宽度和高度上 *总共* 需要填充的像素量
    dw = new_shape[1] - new_un_pad[0]
    dh = new_shape[0] - new_un_pad[1]

    if auto:
        # 在 'auto' 模式下 (通常用于训练):
        # 我们希望填充量是 `stride` (例如 32) 的最小倍数。
        # 这样做可以确保特征图的尺寸能被模型的最大步长整除。
        dw = np.mod(dw, stride)
        dh = np.mod(dh, stride)
    elif scaleFill:
        # 在 'scaleFill' 模式下 (拉伸):
        # 我们根本不需要填充，dw 和 dh 都设为 0。
        # 图像将在下一步被cv2.resize拉伸。
        dw = 0
        dh = 0
        r = (new_shape[1] / shape[1], new_shape[0] / shape[0])
        new_un_pad = (new_shape[1], new_shape[0])

    # 将总填充量除以 2，得到每一侧的填充量
    dw = dw / 2
    dh = dh / 2

    # ==================================================================
    # 3. 调整尺寸 (Resize)
    # ==================================================================

    # 只有当缩放后的尺寸 (new_un_pad) 与原始尺寸 (shape) 不同时，
    # 才执行 cv2.resize。如果尺寸相同，则跳过以节省计算。
    if shape[::-1] != new_un_pad:
        img = cv2.resize(img, new_un_pad, interpolation = cv2.INTER_LINEAR)

    # ==================================================================
    # 4. 应用边框填充 (Pad)
    # ==================================================================
    # 计算上、下、左、右的填充像素数 (必须是整数)
    # 我们使用 round 来四舍五入
    # (dh - 0.1) 和 (dh + 0.1) 是一种处理浮点数不精确问题的小技巧
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    # 使用 cv2.copyMakeBorder 添加边框
    img = cv2.copyMakeBorder(img,top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # ==================================================================
    # 5. 返回结果
    # ==================================================================
    return img, r, (dw * 2, dh * 2)



def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

def xyxy2xywhn(xyxy, w, h, clip=False, eps=1E-3):
    """
    将像素坐标 (xyxy) 转换为归一化的 YOLO 格式坐标 (xywhn)。
    """
    xywhn = np.zeros_like(xyxy)
    x_min, y_min = xyxy[:, 0], xyxy[:, 1]
    x_max, y_max = xyxy[:, 2], xyxy[:, 3]
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    xywhn[:, 0] = x_center / (w + eps)
    xywhn[:, 1] = y_center / (h + eps)
    xywhn[:, 2] = width / (w + eps)
    xywhn[:, 3] = height / (h + eps)
    if clip:
        # np.clip 函数会强制将 xywhn 数组中的所有数字都限制在 [0.0, 1.0] 这个范围内。
        np.clip(xywhn, 0.0, 1.0, out=xywhn)
    return xywhn


def mixup(image, bboxes1, class_labels1, image2, bboxes2, class_labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    image = (image * r + image2 * (1 - r)).astype(np.uint8)
    bboxes1 = np.concatenate((bboxes1, bboxes2), 0)
    class_labels1 = np.concatenate((class_labels1, class_labels2), 0)
    return image, bboxes1, class_labels1

