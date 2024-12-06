import os
import torch  # 导入PyTorch库，用于深度学习模型的实现和训练。
import torch.nn as nn  # 导入PyTorch的神经网络模块，包含神经网络层和损失函数。
import torch.nn.functional as F  # 导入PyTorch的函数式接口，用于定义神经网络的激活函数等。
import torch.optim as optim  # 导入PyTorch的优化器模块，用于优化神经网络参数。
import torchvision  # 导入torchvision库，包含数据集和预训练模型等。
from torchvision import transforms  # 导入torchvision的变换模块，用于数据增强和预处理。
import pytorch_lightning as pl  # 导入PyTorch Lightning库，用于构建深度学习模型。
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging  # 导入logging模块，用于日志记录。
import time
from .clip_voters import *
from clip import clip
from .get_classes import *


class MInterface(pl.LightningModule):
    def __init__(self, args, dtype=torch.float32) -> None:
        super().__init__()
        '''
        args:
            --arch: CLIP模型的名称
            --pretrained_dir: 预训练模型的下载路径
            --logit_scale: logits的缩放因子
            --thres_type: 阈值类型
            --num_classes: 类别数
            --global_thres: 全局阈值 常数
            # -- logit_temperature: logits的温度
            -- ratio: 图像特征的融合比例
            
            -- dataset: 数据集
            
            -- clip_loss: clip损失的权重
            -- alpha_ova: ova损失的权重
            -- weight_freq: clip损失权重更新频率
            # -- device: 设备
        '''
        self.args = args
        self.model = CLIP_Voter(arch=args.arch, pretrained_dir=args.pretrained_dir, logit_scale=args.logit_scale, thres_type=args.thres_type, num_classes=args.num_classes, ratio=args.ratio)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_ova = nn.BCEWithLogitsLoss()
        self.outputs_list = []
        self.classes, self.nums_classes = get_classes(self.args.dataset)
        self.classes = [f"a photo of a {c}" for c in self.classes]
        self.save_hyperparameters(args)
        self.automatic_optimization = True  # 使用自动优化
        self.start_time = 0
        self.end_time = 0 
        

    def forward(self, image_inputs, text_inputs):
        '''
        image_inputs: processed (batch_size, C, H, W)
        text_inputs: processed, (num_classes, D)
        '''
        return self.model(image_inputs = image_inputs, text_inputs = text_inputs)

    def on_train_epoch_start(self):
        # self.model.temperature_scale = torch.min(torch.tensor((self.current_epoch /self.args.temp_warm_epoch * self.args.temp, self.args.temp)))  # 使用最小温度值进行温度升温。
        self.start_time = time.time()  # 记录训练开始时间。
        self.train_loss = 0
        self.train_acc = 0
        self.train_ova_loss = 0
        self.train_reg_loss = 0
        self.train_n = 0 
        self.print(f"第 {self.current_epoch} 轮开始训练：")
        
    
    def training_step(self, batch, batch_idx):
        '''
        每个batch的处理函数
        return :
        Tensor - The loss tensor
        dict - A dictionary. Can include any keys, but must include the key 'loss'
        None - Training will skip to the next batch 
        '''
        with torch.amp.autocast('cuda', enabled=True):
            data, target = batch
            batch_size = len(target)
            label = target.clone().detach().long().view(batch_size, 1)
            one_hot = torch.zeros(batch_size, self.nums_classes, device=label.device).scatter_(1, label, 1)
            text_inputs = clip.tokenize(self.classes).to(label.device) # (num_classes, 77)
            
            logits_voters, logits_per_image = self(image_inputs = data, text_inputs = text_inputs) # logits (batch_size, num_classes)
            logits = logits_voters - self.model.rejection_threshold
            
            # calculating OVA loss
            ova_loss = self.criterion_ova(logits, one_hot) * self.nums_classes

            # calculating CE loss derived from Dempster–Shafer theory of evicence
            zeros = torch.zeros(logits.shape[0], 1).to(label.device)
            logits_ce = torch.cat((logits, zeros), dim=1)
            ce_loss = self.criterion(logits_ce, target)
        
            # calculating clip loss
            clip_loss = self.criterion(logits_per_image, target)
            
            # 根据当前训练阶段动态调整正则化强度
            reg_weight = self._get_regularization_weight(self.current_epoch)
        
            # 组合损失
            main_loss = self.args.alpha_ova * ova_loss + (1 - self.args.alpha_ova) * ce_loss
            if self.args.dataset == 'cifar-10' and self.args.thres_type == 'multi':
                main_loss = main_loss / self.nums_classes
            
            # 最终损失 = 主任务损失 + λ * 正则化损失
            loss = main_loss + reg_weight * clip_loss     
                    
        # 记录结果
        result = {
            "loss": loss.detach(),
            "ova_loss": ova_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "clip_loss": clip_loss.detach(),
            "reg_weight": reg_weight,
            "logits": logits.detach(),  # 不使用 .item()，保持张量
            "target": target,
            "batch_size": batch_size
        }
        self.outputs_list.append(result)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        for output in self.outputs_list:
            self.train_loss += output["loss"] * output["batch_size"]
            self.train_ova_loss += output["ova_loss"] * output["batch_size"] 
            self.train_reg_loss += output["clip_loss"] * output["batch_size"]
            self.train_acc += (output["logits"][:output["batch_size"]].max(1)[1] == output["target"]).sum().item()
            self.train_n += output["batch_size"]           
        self.end_time = time.time()
        self.print(f"Epoch {self.current_epoch} Training finished!!!")
        self.outputs_list.clear()
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        log_string = (
            f"Epoch:            {self.current_epoch}\n"
            f"Train Time:       {self.end_time - self.start_time:.2f}s\n"
            f"Learning Rate:    {current_lr:.6f}\n"
            f"Train Loss:       {self.train_loss / self.train_n:.4f}\n"
            f"Train Ova Loss:   {self.train_ova_loss / self.train_n:.4f}\n"
            f"Train Reg Loss:   {self.train_reg_loss / self.train_n:.4f}\n"
            f"Train Acc:        {self.train_acc / self.train_n:.4f}"
        )
        self.logger.experiment.add_text('train_log', log_string, self.current_epoch)
        self.print(log_string)


    def on_validation_start(self):
        self.validate_n = 0
        self.validate_loss = 0
        self.validate_correct = 0
        self.validate_start_time = time.time()

    def validation_step(self, batch, batch_idx):
        data, target = batch
        batch_size = len(target)
        label = target.clone().detach().long().view(batch_size, 1)
        text_inputs = clip.tokenize(self.classes).to(label.device) # (num_classes, 77)
        
        logits_voters, logits_per_image = self(image_inputs = data, text_inputs = text_inputs) # logits (batch_size, num_classes)
        
        # logits = -(logits_classifier - self.model.rejection_threshold) * self.model.temperature_scale
        logits = logits_voters - self.model.rejection_threshold

        self.validate_correct += (logits.max(1)[1] == label.view(-1)).sum().item()
        
        # 累积正确预测的数量和总样本数
        self.validate_n += batch_size
        self.log('val_acc', self.validate_correct / self.validate_n, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        

    def on_validation_end(self):
        self.validate_end_time = time.time()
        val_acc = self.validate_correct / self.validate_n
        log_string = (
            f"Epoch:           {self.current_epoch}\n"
            f"Validate Time:   {self.validate_end_time - self.validate_start_time:.2f}s\n"
            f"Validate Correct:{self.validate_correct}\n"
            f"Validate Size:   {self.validate_n}\n"
            f"Validate Acc:    {val_acc:.4f}"
        )
        self.logger.experiment.add_text('val_log', log_string, self.current_epoch)
        self.print(log_string)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # 定义余弦退火调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,  # 完整的余弦周期
            eta_min=1e-6  # 最小学习率
        )
        
        # # 定义余弦退火 + 热重启调度器
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        # optimizer,
        # T_0=20,      # 第一次重启的周期（单位：epoch）
        # T_mult=2,    # 每次重启周期倍增（2 倍增长）
        # eta_min=1e-6 # 最小学习率
        # )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # 监控验证损失
                "interval": "epoch",     # 每个epoch更新一次
                "frequency": 1
            }
        }
        
    def _get_regularization_weight(self, epoch):
        """
        动态调整正则化权重的函数
        可以根据训练阶段调整ITM正则化的强度
        """
        base_weight = self.args.clip_loss
        
        # 在训练初期给予较大的正则化权重，随着训练进行逐渐降低
        decay_rate = 0.9
        min_weight = 0.01 * base_weight
        
        weight = max(base_weight * (decay_rate ** (epoch // self.args.weight_freq)), min_weight)
        return weight