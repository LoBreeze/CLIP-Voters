import numpy as np
import sys
from PIL import Image as PILImage
import torch
from torchvision import transforms
import torchvision.transforms as trn
import torchvision.datasets as datasets
import torch.nn.functional as F
import sys
import os
os.chdir('/home/yuantongxin/whx/BCLIP2')
sys.path.append('/home/yuantongxin/whx/BCLIP2/models')
from models.get_classes import get_classes
from models.model_inter_ova import MInterface
import argparse
from train.train_utils import dataset_to_dataloader
from eval_utils import *
from scores_function  import *
from scipy import stats


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # 或者 'max_split_size_mb:32'
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')  # 检测GPU是否可用
kwargs = {'num_workers': 4, 'pin_memory': True}
parser = argparse.ArgumentParser(description='OOD Detection')

model_options = ['RN50','RN101','RN50x4','RN50x16','RN50x64','ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']
dataset_options = ['cifar-10', 'cifar-100',
                   'cub200', 'food101', 'stanford_cars', 'oxford_iiit_pet',
                   'imagenet20', 'imagenet10', 'imagenet100', 'imagenet200']  # 支持的数据集选项列表。
score_options = ['energy', 'max-logit', 'msp', 'odin', 'mahalanobis','ova']
# 基础的测试设置
parser.add_argument('--test-batch-size', type=int,  default=128, metavar='N', help='input batch size for testing (default: 128)')  # 测试批次大小。
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--dataset', default='imagenet', choices=dataset_options)  # 选择数据集。
parser.add_argument('--data-dir', default='/home/yuantongxin/whx/BCLIP2/data', help='directory of dataset for testing')  # 数据集的存储路径。
parser.add_argument('--model', '-a', default='ViT-B/32', choices=model_options)
parser.add_argument('--checkpoint_path', '-c', default=None, type=str, help='model path')  # 模型路径。
parser.add_argument('--score', default='ova', type=str, help='score type', choices=score_options)  # 分数类型。
parser.add_argument('--T', type=float, default=1, help='score temperature parameter') # It is better to set T to 0.01 for energy score in our framework
parser.add_argument('--logs', default='/Users/utopia/Documents/blip2/project/comparison/results', type=str, help='log directory')  # 日志目录。
parser.add_argument('--ensemble', action='store_true', default=False, help='CLIP text prompt engineering')
args = parser.parse_args()  # 解析命令行参数


kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}  # 设置数据加载器参数
classes, num_classes = get_classes(args.dataset)  # 获取数据集的类别列表和类别数。
setattr(args, 'num_classes', num_classes)  # 添加新的属性
torch.manual_seed(args.seed)  # 设置随机种子。
os.makedirs(args.logs, exist_ok=True)  # 创建日志目录

model = MInterface.load_from_checkpoint(checkpoint_path=args.checkpoint_path, map_location=torch.device('cpu')).to(device)  # 加载模型
model.eval()  # 设置为评估模式

train_loader, test_loader = dataset_to_dataloader(args.dataset, args.data_dir, args.test_batch_size, **kwargs)  # 获取数据加载器
del train_loader  # 删除训练加载器

# /////////////// OOD检测准备 ///////////////
auroc_list, aupr_list, fpr_list = [], [], []  # 初始化AUC和FPR列表
log = setup_log(args)


in_score, right_score, wrong_score, list_softmax_ID, list_correct_ID = get_ood_scores(args=args, loader=test_loader, model=model, device=device, in_dist=True)  # 获取ID数据集的分数
log.debug(f"{'='*10} {args.dataset} {'='*10}")
num_right = len(right_score)  # 正确样本数量
num_wrong = len(wrong_score)  # 错误样本数量
test_accuracy = 100 * num_right / (num_wrong + num_right)  # 计算测试准确率
log.debug("Accuracy testing ...")
log.debug(f"Test Accuracy: {test_accuracy:.2f}%")  # 打印测试准确率

# 计算AURC和EAURC
# right_score: _scores[right_indices], wrong_score: _scores[wrong_indices]
log.debug("Misclassification detection ...")
measures = get_measures(-right_score, -wrong_score)  # 性能
print_measures(log, measures[0], measures[1], measures[2], method_name=args.dataset)  # 打印性能指标
ID_aurc, ID_eaurc = calc_aurc_eaurc(list_softmax_ID, list_correct_ID)  # 计算AURC和EAURC
log.debug(f"ID AURC: {ID_aurc:.2f}, ID EAURC: {ID_eaurc:.2f}")  # 打印AURC和EAURC

log.debug(f"OOD Detection ...")
log.debug(f"in scores: {in_score[:5]}")
log.debug(f"in scores: {stats.describe(in_score)}")  # 打印ID数据集的分数
# 获得ood数据集
ood_lists = get_id_ood_datasets(args.dataset)

for ood in ood_lists:
    log.debug(f"Evaluting OOD dataset {ood}")
    ood_loader = set_ood_loader(dataset_name=ood, root_dir=args.data_dir, batch_size=args.test_batch_size, **kwargs)  # 获取OOD数据加载器
    out_score, list_softmax_OOD, list_correct_OOD = get_ood_scores(args=args, loader=ood_loader, model=model, device=device)  # 获取OOD数据集的分数
    log.debug(f"in scores: {in_score[:5]}")
    log.debug(f"out scores: {out_score[:5]}")
    log.debug(f"out scores: {stats.describe(out_score)}")  # 打印OOD数据集的分数
    get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list)
    
log.debug(f'\n\n{args.score} Mean Test Results')
print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                np.mean(fpr_list), method_name=args.score)

save_as_dataframe(args, ood_lists, fpr_list, auroc_list, aupr_list)