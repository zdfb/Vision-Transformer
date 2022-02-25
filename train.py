import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import utils.dataloader as DataSet
import torch.backends.cudnn as cudnn

from nets.vit import vit_base_patch16_384
from utils.utils_fit import fit_one_epoch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


###### 训练vit模型 ######


class train_vit():
    def __init__(self, num_classes):
        super(train_vit, self).__init__()

        model_path = 'vit_base_patch16_384.pth'
        
        self.train_txt_path = 'main_txt/train.txt'  # 存放训练集的txt文件路径
        self.test_txt_path = 'main_txt/test.txt'  # 存放测试集的txt文件路径

        # 创建vit模型
        model = vit_base_patch16_384(model_path)
        inchannel = model.head.in_features
        model.head = nn.Linear(inchannel, num_classes)  # 修改输出类别
        
        if torch.cuda.is_available():
            cudnn.benchmark = True
            model = model.to(device)
        self.model = model

        self.vit_loss = nn.CrossEntropyLoss()  # 定义损失函数
        self.test_accuracy_max = 0  # 初始化最大测试集准确率 

    def train(self, batch_size, epochs):

        # 使用分组学习率策略，分类器部分学习率较高，特征提取部分学习率较低
        high_rate_params = []
        low_rate_params = []

        for name, params in self.model.named_parameters():
            if 'head' in name:
                high_rate_params += [params]
            else:
                low_rate_params += [params]
        
        optimizer = optim.SGD(
            params=[
                {"params": high_rate_params, 'lr': 0.001},
                {"params": low_rate_params},
            ],
            lr=0.0002, momentum=0.8, weight_decay=5e-4)
        
        # 学习率下降策略
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.3, last_epoch = -1)

        # 定义训练集与测试集
        train_data = torch.utils.data.DataLoader(DataSet.train_dataset(self.train_txt_path), batch_size = batch_size, shuffle = True, num_workers = 2)

        test_data = torch.utils.data.DataLoader(DataSet.test_dataset(self.test_txt_path), batch_size = batch_size, shuffle = False, num_workers = 2)
        
        # 开始训练
        for epoch in range(epochs):
            print('Epoch: ', epoch)
            train_accuracy, test_accuracy = fit_one_epoch(self.model, self.vit_loss, optimizer, train_data, test_data, device)
            lr_scheduler.step()

            if test_accuracy > self.test_accuracy_max:
                self.test_accuracy_max = test_accuracy
                torch.save(self.model.state_dict(), 'vit.pth')

if __name__ == "__main__":
    train = train_vit(1000)
    train.train(1000, 50)