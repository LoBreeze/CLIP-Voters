import pytorch_lightning as pl  # pytorch-lightning       2.3.0
from lightning.pytorch import Trainer
from lightning.pytorch.tuner import Tuner  # 仅导入 Tuner

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
import argparse
import os
from datetime import datetime

import sys
sys.path.append('/Users/utopia/Documents/blip2/voters')
from models.model_inter_ova import MInterface
from models.get_classes import get_classes
from train_utils import dataset_to_dataloader

# 设置可选的环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # 或者 'max_split_size_mb:32'
devices = [2,3,4,5]

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training for BLIP2_Adapter')
model_options = ['RN50','RN101','RN50x4','RN50x16','RN50x64','ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']
dataset_options = ['cifar-10', 'cifar-100',
                   'cub200', 'food101', 'stanford_cars', 'oxford_iiit_pet',
                   'imagenet20', 'imagenet10', 'imagenet100', 'imagenet200']  # 支持的数据集选项列表。
threshold_options = ['one', 'multi','const']  # 阈值类型选项列表。
# 基础的训练设置
parser.add_argument('--train-batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 512)')  # 训练批次大小。
parser.add_argument('--test-batch-size', type=int,  default=32, metavar='N', help='input batch size for testing (default: 512)')  # 测试批次大小。
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')  # 训练轮数。
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--lr_max', type=float, default=None, metavar='LR', help='learning rate')

# 模型的选择
parser.add_argument('--model_name', '-a', default='ViT-B/32', choices=model_options)
parser.add_argument('--pretrained_dir', type=str, default='~/.cache/clip', help='预训练模型的下载路径')
parser.add_argument('--logit_scale', type=float, default=None, help='logits的缩放因子')
parser.add_argument('--thres_type', type=str, default='one', choices=threshold_options, help='阈值类型')
parser.add_argument('--ratio', type=float, default=0.2, help='图像特征的融合比例')
parser.add_argument('--clip_loss', type=float, default=1.0, help='clip损失的权重')
parser.add_argument('--alpha_ova', type=float, default=0.9, help='ova损失的权重')
parser.add_argument('--weight_freq', type=int, default=10, help='clip损失权重更新频率')

# 数据集的选择
parser.add_argument('--dataset', '-d', default='cifar-10', choices=dataset_options)  # 选择数据集。
parser.add_argument('--data-dir', default='/Users/utopia/Documents/blip2/voters/data', help='directory of dataset for training and testing')  # 数据集的存储路径。

# 模型保存目录和日志目录
parser.add_argument('--save-dir', default='/Users/utopia/Documents/blip2/voters/train/checkpoints', type=str, help='directory to save logs and models')
#相关设置
parser.add_argument('--num_workers', type=int, default=1, metavar='LR', help='number of workers for data loading')  # 数据加载时的工作线程数。
# 解析命令行参数
args = parser.parse_args()   

kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
classes, num_classes = get_classes(args.dataset)  # 获取数据集的类别列表和类别数。
setattr(args, 'num_classes', num_classes)  # 添加新的属性
if args.thres_type == 'const':
    setattr(args, 'global_thres', 150)  # 添加新的属性
    
save_dir = os.path.join(args.save_dir, args.dataset, args.model_name)
os.makedirs(save_dir, exist_ok=True)
checkpoint_path = os.path.join(save_dir, 'lr_rate_'+ str(args.lr_max))
logger_path = os.path.join(save_dir, 'lr_rate_'+ str(args.lr_max))

# 定义回调函数
callbacks = [
    # 模型检查点
    ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_path,
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True  # 保存最新的检查点
    ),
    # 学习率监视器
    LearningRateMonitor(logging_interval='step'),
]

# logger
current_time = datetime.now().strftime('%m-%d_%H-%M')
logger = pl_loggers.TensorBoardLogger(
    save_dir=logger_path,
    version=current_time,
    name=f'{args.dataset}_training'
)

torch.manual_seed(args.seed)  # 设置随机种子。

# 初始化模型
model = MInterface(args=args)

# 加载数据
train_loader, test_loader = dataset_to_dataloader(args.dataset, args.data_dir, args.train_batch_size, args.test_batch_size, **kwargs)

# 启用 TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
# 使用 CuDNN 动态优化卷积操作
torch.backends.cudnn.benchmark = True
# 允许非确定性算法以提高性能
torch.backends.cudnn.deterministic = False


# 使用 Trainer 进行训练
# trainer = pl.Trainer(
#     max_epochs=args.epochs,  # 最大训练轮数
#     accelerator='gpu',  # 分布式数据并行 (DDP)
#     devices= devices ,  # 使用 2 个 GPU 进行训练 'auto', [1,2], 7
#     strategy='ddp_find_unused_parameters_true',  # 分布式数据并行 (DDP) 策略
#     logger=logger,  # 使用 TensorBoard 记录
#     callbacks=callbacks,  # 模型保存回调
#     enable_progress_bar=True,
#     accumulate_grad_batches=16,  
#     # gradient_clip_val=1.0,
#     # limit_train_batches=1.0,
# )


# mac版本
trainer = pl.Trainer(
    max_epochs=args.epochs,  # 最大训练轮数
    accelerator='mps',  # 使用 MPS 加速
    devices=1,  # 使用单个 GPU
    logger=logger,  # 使用 TensorBoard 记录
    callbacks=callbacks,  # 模型保存回调
    enable_progress_bar=True,
    accumulate_grad_batches=16,  
    # gradient_clip_val=1.0,
    # limit_train_batches=1.0,
)


# 学习率设置
new_lr = args.lr_max
if new_lr is None:
    try:
        # 使用新的 Tuner API
        tuner = Tuner(Trainer())
        # 运行学习率查找
        lr_finder = tuner.lr_find(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=test_loader,
            max_lr=1.0,
            min_lr=1e-8,
            num_training=100,
        )
        # Results can be found in
        print(lr_finder.results)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()
        # 获取建议的学习率
        new_lr = lr_finder.suggestion()
        print(f'建议的学习率: {new_lr}')
        
    except Exception as e:
        print(f'学习率查找失败: {str(e)}')
        print('使用默认学习率 1e-4')
        new_lr = 1e-4
    exit()

# 更新模型的学习率
model.hparams.learning_rate = new_lr
print(f'Using learning rate: {model.hparams.learning_rate}')

# 开始训练
print('Start training...')
trainer.fit(model, train_loader, test_loader)
