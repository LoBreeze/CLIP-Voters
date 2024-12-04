import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import os.path
import six
import random
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class LSUNClass(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        import lmdb  # 导入 lmdb 库，用于处理 LMDB 数据库
        self.db_path = db_path  # 保存数据库路径
        # 打开 LMDB 数据库，设置为只读模式
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        # 获取数据库中的条目数量
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        # 创建缓存文件路径
        cache_file = '_cache_' + db_path.replace('/', '_')
        # 如果缓存文件存在，加载键
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            # 否则，从数据库中读取所有键并缓存
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))
        self.transform = transform  # 保存图像转换函数
        self.target_transform = target_transform  # 保存目标转换函数

    def __getitem__(self, index):
        img, target = None, None  # 初始化图像和目标
        env = self.env  # 获取数据库环境
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])  # 获取指定索引的图像数据

        buf = six.BytesIO()  # 创建一个字节缓冲区
        buf.write(imgbuf)  # 将图像数据写入缓冲区
        buf.seek(0)  # 移动到缓冲区的开头
        img = Image.open(buf).convert('RGB')  # 打开图像并转换为 RGB 格式

        if self.transform is not None:
            img = self.transform(img)  # 应用图像转换

        if self.target_transform is not None:
            target = self.target_transform(target)  # 应用目标转换

        return img, target  # 返回图像和目标

    def __len__(self):
        return self.length  # 返回数据集长度

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'  # 返回类的字符串表示

class LSUN(data.Dataset):
    """
    `LSUN <http://lsun.cs.princeton.edu>`_ 数据集。
    参数:
        db_path (string): 数据库文件的根目录。
        classes (string or list): 'train', 'val', 'test'之一或要加载的类别列表。
        transform (callable, optional): 一个函数/转换，接受 PIL 图像并返回转换后的版本。
        target_transform (callable, optional): 一个函数/转换，接受目标并进行转换。
    """
    def __init__(self, db_path, classes='test',
                 transform=None, target_transform=None):
        # 定义可用类别
        categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                      'conference_room', 'dining_room', 'kitchen',
                      'living_room', 'restaurant', 'tower']
        dset_opts = ['train', 'val', 'test']  # 数据集选项
        self.db_path = db_path  # 保存数据库路径
        # 如果 classes 是字符串且在选项中
        if type(classes) == str and classes in dset_opts:
            if classes == 'test':
                classes = [classes]  # 测试集
            else:
                classes = [c + '_' + classes for c in categories]  # 添加类别前缀
        self.classes = classes  # 保存类列表

        # 为每个类创建一个 LSUNClass 数据集
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(
                db_path=db_path + '/' + c + '_lmdb',
                transform=transform))  # 初始化 LSUNClass 对象

        self.indices = []  # 存储每个类的索引
        count = 0  # 初始化计数器
        for db in self.dbs:
            count += len(db)  # 累计数据集长度
            self.indices.append(count)  # 添加到索引列表

        self.length = count  # 总数据集长度
        self.target_transform = target_transform  # 保存目标转换函数

    def __getitem__(self, index):
        """
        参数:
            index (int): 索引
        返回:
            tuple: 包含图像和目标的元组，其中目标是目标类别的索引。
        """
        target = 0  # 初始化目标类别
        sub = 0  # 初始化子索引
        for ind in self.indices:  # 遍历索引
            if index < ind:  # 找到目标类
                break
            target += 1  # 更新目标类索引
            sub = ind  # 更新子索引

        db = self.dbs[target]  # 获取对应的 LSUNClass 数据集
        index = index - sub  # 调整索引

        if self.target_transform is not None:
            target = self.target_transform(target)  # 应用目标转换

        img, _ = db[index]  # 从子数据集中获取图像
        return img, target  # 返回图像和目标

    def __len__(self):
        return self.length  # 返回数据集长度

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'  # 返回类的字符串表示


if __name__ == '__main__':
    # 创建 LSUN 数据集对象
    lsun = LSUN(db_path='/home/yuantongxin/whx/CIFAR/data/odd_data/LSUN', transform=transforms.Compose([transforms.ToTensor()]))
    print(f"Total number of samples in the LSUN dataset: {len(lsun)}")
    for index in range(20):
        i = random.randint(0, len(lsun))
        print(f"Index: {i}")
        print(f"Label: {lsun[i][1]}")
        print(f"Image shape: {lsun[i][0].shape}")