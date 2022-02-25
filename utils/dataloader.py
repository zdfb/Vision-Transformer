import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


###### 定义数据读取 ######


# 将输入图像转化为RGB形式
def default_loader(path):
    return Image.open(path).convert('RGB')

# 定义数据读取类
class Vit_dataset(Dataset):
    def __init__(self, txt, transform = None, loader = default_loader):
        super(Vit_dataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除本行末尾的分行字符
            words = line.split()  # 分割图片路径及标签
            imgs.append((words[0], int(words[1])))
        
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
    
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.imgs)

# 定义预处理
data_transform = {
    "train": transforms.Compose([
        transforms.Resize(438),
        transforms.RandomResizedCrop(384),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    "test": transforms.Compose([transforms.Resize(438),
                               transforms.CenterCrop(384),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

def train_dataset(root):
    return Vit_dataset(root, transform = data_transform["train"])

def test_dataset(root):
    return Vit_dataset(root, transform = data_transform["test"])