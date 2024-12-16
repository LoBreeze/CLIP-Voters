import sys
import os
# 获取当前脚本的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取上一层目录路径
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
# 将上一层目录添加到 sys.path
sys.path.append(parent_dir)

from eval_utils import *
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(root_dir, 'data')

id_dataset =StanfordCars98(root=root_dir, split='train', id=True, mode='id')
test = OxfordIIITPet_18(root=root_dir, split='trainval', id=False, mode='ood')