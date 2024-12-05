import pytorch_lightning as pl  # pytorch-lightning       1.9.0
from pytorch_lightning.strategies import DDPStrategy  # 分布式数据并行 (DDP) 策略
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
import argparse
from datetime import datetime
import sys
import os
# 获取当前脚本的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取上一层目录路径
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
# 将上一层目录添加到 sys.path
sys.path.append(parent_dir)
from models.model_inter_ova import MInterface
from models.get_classes import get_classes
from train_utils import *

# 设置可选的环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # 或者 'max_split_size_mb:32'
devices = [2,3,4,5]
tmp_dir = os.path.join(parent_dir, 'cache')

def get_args():
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
    parser.add_argument('--learning_rate', type=float, default=0.003311, metavar='LR', help='learning rate')

    # 模型的选择
    parser.add_argument('--arch', default='ViT-B/32', choices=model_options)
    parser.add_argument('--pretrained_dir', type=str, default= tmp_dir, help='预训练模型的下载路径')
    parser.add_argument('--logit_scale', type=float, default=None, help='logits的缩放因子')
    parser.add_argument('--thres_type', type=str, default='one', choices=threshold_options, help='阈值类型')
    parser.add_argument('--ratio', type=float, default=0.2, help='图像特征的融合比例')
    parser.add_argument('--clip_loss', type=float, default=1.0, help='clip损失的权重')
    parser.add_argument('--alpha_ova', type=float, default=0.9, help='ova损失的权重')
    parser.add_argument('--weight_freq', type=int, default=10, help='clip损失权重更新频率')

    # 数据集的选择
    parser.add_argument('--dataset', '-d', default='cifar-10', choices=dataset_options)  # 选择数据集。
    parser.add_argument('--data-dir', default=os.path.join(parent_dir, 'data') , help='directory of dataset for training and testing')  # 数据集的存储路径。

    # 模型保存目录和日志目录
    parser.add_argument('--save-dir', default=os.path.join(parent_dir, 'train', 'checkpoint') , type=str, help='directory to save logs and models')
    #相关设置
    parser.add_argument('--num_workers', type=int, default=2, metavar='LR', help='number of workers for data loading')  # 数据加载时的工作线程数。
    parser.add_argument('--auto-find', type=bool, default=True, help='automatically find best lr and batch_size')  # 数据加载时的工作线程数。
    # 解析命令行参数
    args = parser.parse_args()  
    return args

def train(args): 
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    classes, num_classes = get_classes(args.dataset)  # 获取数据集的类别列表和类别数。
    setattr(args, 'num_classes', num_classes)  # 添加新的属性
    save_dir = os.path.join(args.save_dir, args.dataset, args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, 'lr_rate_'+ str(args.learning_rate))
    logger_path = os.path.join(save_dir, 'lr_rate_'+ str(args.learning_rate))
    
    # 定义回调函数
    callbacks = [
        # 模型检查点
        ModelCheckpoint(
            monitor='val_acc',
            dirpath=checkpoint_path,
            filename='model-{epoch:02d}-{val_acc:.2f}',
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
    # print(model.hparams)

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
    #     # strategy='ddp_find_unused_parameters_true',  # 分布式数据并行 (DDP) 策略
    #     strategy=DDPStrategy(find_unused_parameters=True),  # 分布式数据并行 (DDP) 策略  | None
    #     logger=logger,  # 使用 TensorBoard 记录
    #     callbacks=callbacks,  # 模型保存回调
    #     enable_progress_bar=True,
    #     accumulate_grad_batches=16,
    #     default_root_dir = tmp_dir, # 保存一些其他文件  
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
        default_root_dir = tmp_dir, # 保存一些其他文件
        # gradient_clip_val=1.0,
        # limit_train_batches=1.0,
    )
    
    if args.auto_find:
        suggest_lr = auto_find_lr(trainer, model, train_loader, test_loader)
        print(f'Suggested learning rate: {suggest_lr}')
        print('Please set the learning rate in the args and run the script again.')
        exit(0)

    # 开始训练
    print('\n\nStart training...')
    trainer.fit(model, train_loader, test_loader)


if __name__ == '__main__':
    args = get_args()
    train(args)