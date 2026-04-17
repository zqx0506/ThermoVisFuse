import os
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np

# ===========================
# 数据增强函数（VT5000 版）
# ===========================
def cv_random_flip(img, t, gt):
    if random.randint(0, 1):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        t   = t.transpose(Image.FLIP_LEFT_RIGHT)
        gt  = gt.transpose(Image.FLIP_LEFT_RIGHT)
    return img, t, gt

def randomCrop(img, t, gt, border=30):
    w, h = img.size
    crop_w = np.random.randint(w - border, w)
    crop_h = np.random.randint(h - border, h)
    region = ((w - crop_w) >> 1, (h - crop_h) >> 1, (w + crop_w) >> 1, (h + crop_h) >> 1)
    return img.crop(region), t.crop(region), gt.crop(region)

def randomRotation(img, t, gt):
    if random.random() > 0.8:
        angle = np.random.randint(-15, 15)
        img = img.rotate(angle, Image.BICUBIC)
        t   = t.rotate(angle, Image.BICUBIC)
        gt  = gt.rotate(angle, Image.BICUBIC)
    return img, t, gt

def colorEnhance(image):
    bright_intensity = random.randint(5, 15)/10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15)/10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20)/10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30)/10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0]-1)
        randY = random.randint(0, img.shape[1]-1)
        img[randX, randY] = 0 if random.randint(0,1) == 0 else 255
    return Image.fromarray(img)

# ===========================
# 训练集
# ===========================
class SalObjDataset(data.Dataset):
    def __init__(self, train_root, trainsize):
        self.trainsize = trainsize

        self.image_root  = train_root + '/RGB/'
        self.gt_root     = train_root + '/GT/'
        self.t_root      = train_root + '/T/'

        # VT5000 没有 body 和 Edge
        self.body_root   = None
        self.detail_root = None

        self.images = sorted([self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts    = sorted([self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.png') or f.endswith('.jpg')])
        self.ts     = sorted([self.t_root + f for f in os.listdir(self.t_root) if f.endswith('.jpg') or f.endswith('.png')])

        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.525, 0.590, 0.537], [0.177, 0.167, 0.176])
        ])
        self.t_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.736, 0.346, 0.339], [0.179, 0.196, 0.169])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image  = self.rgb_loader(self.images[index])
        t      = self.rgb_loader(self.ts[index])
        gt     = self.binary_loader(self.gts[index])

        # 数据增强
        image, t, gt = cv_random_flip(image, t, gt)
        image, t, gt = randomCrop(image, t, gt)
        image, t, gt = randomRotation(image, t, gt)

        image = colorEnhance(image)
        t     = colorEnhance(t)
        gt    = randomPeper(gt)

        image = self.img_transform(image)
        t     = self.t_transform(t)
        gt    = self.gt_transform(gt)

        return image, t, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

# ===========================
# 训练 dataloader
# ===========================
def get_loader(train_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=False):
    dataset = SalObjDataset(train_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

# ===========================
# 测试集
# ===========================
class test_dataset:
    def __init__(self, test_root, testsize):
        self.testsize = testsize

        self.image_root = test_root + '/RGB/'
        self.gt_root    = test_root + '/GT/'
        self.t_root     = test_root + '/T/'

        self.images = sorted([self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts    = sorted([self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.png') or f.endswith('.jpg')])
        self.ts     = sorted([self.t_root + f for f in os.listdir(self.t_root) if f.endswith('.jpg') or f.endswith('.png')])

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.525, 0.590, 0.537], [0.177, 0.167, 0.176])
        ])
        self.t_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.736, 0.346, 0.339], [0.179, 0.196, 0.169])
        ])
        self.gt_transform = transforms.ToTensor()

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        shape = image.size
        image = self.img_transform(image).unsqueeze(0)

        t = self.rgb_loader(self.ts[self.index])
        t = self.t_transform(t).unsqueeze(0)

        gt    = self.binary_loader(self.gts[self.index])
        name = os.path.basename(self.images[self.index])
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index = (self.index + 1) % self.size
        return image, t, gt, shape, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
