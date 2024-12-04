from torch.utils.data import Dataset
from torchvision.datasets.utils import check_integrity, download_url
from PIL import Image
import numpy as np
import os

    
class SVHN(Dataset):  # 定义SVHN类，继承自PyTorch的Dataset类
    url = ""  # 数据集下载URL
    filename = ""  # 数据集文件名
    file_md5 = ""  # 数据集文件的MD5校验和

    # 数据集的不同分割信息，包括训练集、测试集、额外数据集和训练加额外数据集
    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",  # 训练集下载链接
                  "train_32x32.mat",  # 训练集文件名
                  "e26dedcc434d2e4c54c9b2d4a06d8373"],  # 训练集文件MD5
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",  # 测试集下载链接
                 "test_32x32.mat",  # 测试集文件名
                 "eb5a983be6a315427106f1b164d9cef3"],  # 测试集文件MD5
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",  # 额外数据集下载链接
                  "extra_32x32.mat",  # 额外数据集文件名
                  "a93ce644f1a588dc4d68dda5feec44a7"],  # 额外数据集文件MD5
        'train_and_extra': [  # 训练和额外数据集组合
                ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",  # 训练集下载链接
                 "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],  # 训练集文件信息
                ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",  # 额外数据集下载链接
                 "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]]  # 额外数据集文件信息
    }

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        self.root = root  # 数据集根目录
        self.transform = transform  # 图像变换操作
        self.target_transform = target_transform  # 标签变换操作
        self.split = split  # 选择的数据集分割（训练、测试、额外或训练加额外）

        # 检查分割是否有效
        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test" '
                             'or split="train_and_extra" ')

        # 根据选择的分割类型设置下载信息
        if self.split == "train_and_extra":
            self.url = self.split_list[split][0][0]  # 设置下载链接
            self.filename = self.split_list[split][0][1]  # 设置文件名
            self.file_md5 = self.split_list[split][0][2]  # 设置MD5
        else:
            self.url = self.split_list[split][0]  # 下载链接
            self.filename = self.split_list[split][1]  # 文件名
            self.file_md5 = self.split_list[split][2]  # MD5

        # 导入scipy.io用于读取.mat文件
        import scipy.io as sio

        # 读取.mat文件并加载数据为数组
        loaded_mat = sio.loadmat(os.path.join(root, self.filename))

        # 处理测试集数据
        if self.split == "test":
            self.data = loaded_mat['X']  # 获取图像数据
            self.targets = loaded_mat['y']  # 获取标签数据
            # 标签10表示数字0，使用取模操作转换为0-9的索引
            self.targets = (self.targets % 10).squeeze()  # 转换为零基索引
            self.data = np.transpose(self.data, (3, 2, 0, 1))  # 转置数据维度
        else:  # 处理训练集和额外数据
            self.data = loaded_mat['X']  # 获取图像数据
            self.targets = loaded_mat['y']  # 获取标签数据

            if self.split == "train_and_extra":  # 如果是训练加额外数据
                extra_filename = self.split_list[split][1][1]  # 额外数据集文件名
                loaded_mat = sio.loadmat(os.path.join(root, extra_filename))  # 加载额外数据集
                self.data = np.concatenate([self.data, loaded_mat['X']], axis=3)  # 合并数据
                self.targets = np.vstack((self.targets, loaded_mat['y']))  # 合并标签
            # 标签10表示数字0，使用取模操作转换为0-9的索引
            self.targets = (self.targets % 10).squeeze()  # 转换为零基索引
            self.data = np.transpose(self.data, (3, 2, 0, 1))  # 转置数据维度

    def __getitem__(self, index):
        # 根据索引获取图像和标签
        img, target = self.data[index], self.targets[index]

        # 将图像转换为PIL图像格式，以保持与其他数据集的一致性
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        # 应用图像变换（如果有的话）
        if self.transform is not None:
            img = self.transform(img)

        # 应用标签变换（如果有的话）
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target  # 返回处理后的图像和标签

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)  # 无论是测试集还是训练集，长度相同

    def _check_integrity(self):
        # 检查数据集文件的完整性
        root = self.root  # 数据集根目录
        if self.split == "train_and_extra":  # 如果是训练加额外数据
            md5 = self.split_list[self.split][0][2]  # 获取训练集MD5
            fpath = os.path.join(root, self.filename)  # 训练集文件路径
            train_integrity = check_integrity(fpath, md5)  # 检查训练集完整性
            extra_filename = self.split_list[self.split][1][1]  # 获取额外数据集文件名
            md5 = self.split_list[self.split][1][2]  # 获取额外数据集MD5
            fpath = os.path.join(root, extra_filename)  # 额外数据集文件路径
            return check_integrity(fpath, md5) and train_integrity  # 检查两个数据集的完整性
        else:  # 对于其他分割
            md5 = self.split_list[self.split][2]  # 获取MD5
            fpath = os.path.join(root, self.filename)  # 文件路径
            return check_integrity(fpath, md5)  # 检查文件完整性

    def download(self):
        # 下载数据集文件
        if self.split == "train_and_extra":  # 如果是训练加额外数据
            md5 = self.split_list[self.split][0][2]  # 获取训练集MD5
            download_url(self.url, self.root, self.filename, md5)  # 下载训练集文件
            extra_filename = self.split_list[self.split][1][1]  # 获取额外数据集文件名
            md5 = self.split_list[self.split][1][2]  # 获取额外数据集MD5
            download_url(self.url, self.root, extra_filename, md5)  # 下载额外数据集文件
        else:  # 对于其他分割
            md5 = self.split_list[self.split][2]  # 获取MD5
            download_url(self.url, self.root, self.filename, md5)  # 下载数据集文件
