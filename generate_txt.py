import os
import random


###### 划分训练集与测试集 ######


# 获取文件的绝对路径
def get_files_path(file_dir):
    dir_names = []
    for fn in os.listdir(file_dir):  # fn表示文件名
        dir_names.append(os.path.join(file_dir, fn))
    return dir_names

# 获得文件夹下所有文件路径
def get_dir_img(file_dir):
    filenames = []
    for root, dirs, files in os.walk(file_dir):
        for name in files:
            filenames.append(os.path.join(root, name))
    return filenames

# 生成表示训练集与数据集的txt文件
def make_txt(img_root, txt_root, quantity_proportion):

    # 创建txt文件
    txt_name = [txt_root + '/train.txt', txt_root + '/test.txt']

    train = open(txt_name[0], 'a')  # 打开用于存储训练样本的txt文件
    test = open(txt_name[1], 'a')  # 打开用于存储测试样本的txt文件

    sort_files = get_files_path(img_root)  # 获取存储图片的类别文件夹名

    for index, file_path in enumerate(sort_files):
        tem_total_img = get_dir_img(file_path)  # 该文件夹下的所有图片路径
        random.shuffle(tem_total_img)

        num_img = len(tem_total_img)  # 该文件夹下的所有图片数量
    
        span_num = [int(x * num_img) for x in quantity_proportion]

        for i in range(span_num[0] + 1):
            train.write(tem_total_img[i] + ' ' + str(index) + '\n')
        for j in range(span_num[0] + 1, num_img):
            test.write(tem_total_img[j] + ' ' + str(index) + '\n')  

if __name__ == '__main__':
    
    img_root = 'JpegImages'
    txt_root = 'main_txt'
    quantity_proportion = [0.9, 0.1]
    make_txt(img_root, txt_root, quantity_proportion)