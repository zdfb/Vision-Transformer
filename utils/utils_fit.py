import time
import torch
import numpy as np


###### 将vit网络训练一个epoch ######


def fit_one_epoch(model, vit_loss, optimizer, train_data, test_data, device):
    start_time = time.time()  # 获取当前时间
    model.train()  # 训练过程

    train_loss_list = []  # 用于记录训练集loss
    train_accuracy_list = []  # 用于记录训练集准确率

    for step, data in enumerate(train_data):
        images, targets = data[0], data[1]  # 取出图片及标签

        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 前向传播

        loss_value = vit_loss(outputs, targets)  # 计算loss值

        loss_value.backward()  # 反向传播
        optimizer.step()  # 优化器迭代

        train_loss_list.append(loss_value.item())

        prediction = torch.max(outputs, dim = 1)[-1]
        train_accuracy = prediction.eq(targets).cpu().float().mean()
        train_accuracy_list.append(train_accuracy)
        
        # 画进度条
        rate = (step + 1) / len(train_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(
            int(rate * 100), a, b, loss_value), end="")
    print()
    print('train_loss:{:.3f},train_accuracy:{:.3f}'.format(np.mean(train_loss_list), np.mean(train_accuracy_list)))

    model.eval()  # 测试过程
    test_loss_list = []
    test_accuracy_list = []

    for step, data in enumerate(test_data):
        images, targets = data[0], data[1]  # 取出图片及标签

        with torch.no_grad():
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()  # 清零梯度
            outputs = model(images)  # 前向传播

            loss_value = vit_loss(outputs, targets)

            test_loss_list.append(loss_value.item)
            prediction = torch.max(outputs, dim = 1)[-1]
            test_accuracy = prediction.eq(targets).cpu().float().mean()
            test_accuracy_list.append(test_accuracy)

        rate = (step + 1) / len(test_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtest loss: {:^3.0f}%[{}->{}]{:.3f}".format(
            int(rate * 100), a, b, loss_value), end="")
    print()
    print('test_loss:{:.3f},test_accuracy:{:.3f}, epoch_time:{:.3f}.'.format(np.mean(test_loss_list), np.mean(test_accuracy_list), time.time() - start_time))
    
    return np.mean(train_accuracy_list), np.mean(test_accuracy_list)