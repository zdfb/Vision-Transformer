# Vision Transformer(ViT)   
Pytorch实现的简单的Vision Transformer(ViT)分类任务。

## 预训练模型
+ ViT_base_patch16_384模型pth格式的预训练权重如下。<br>
>- 链接：https://pan.baidu.com/s/1y1kOvlR9-OUUrZpRSkww3w
>- 提取码：ampe

## 训练自己的数据集
### 1. 按照如下格式准备数据集,文件路径中不允许出现中文或空格。
```bash
--JpegImages
    --category_1
        --001.jpg
        --002.jpg
        ...
    --category_2
        --001.jpg
        --002.jpg
        ...
    ...
```
### 2. 划分训练集与测试集
新建main_txt文件夹，用于存放划分后的训练集与测试集信息。修改generate_txt.py文件中的图片集指向及txt存放位置指向，运行：
``` bash
python generate_txt.py
```
运行完毕后，生成train.txt与test.txt, 存放至main_txt文件夹下。
### 3.  开始训练
修改train.py中的batch_size参数及类别数量，调整学习率及相关参数，运行：
``` bash
python train.py
```
## Reference
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
