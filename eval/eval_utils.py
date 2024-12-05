import logging
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import sklearn.metrics as sk
import logging
from torch.autograd import Variable
import torch.nn as nn
from data_interface.bird200 import *
from data_interface.car196 import *
from data_interface.food101 import *
from data_interface.pet37 import *
from data_interface.lsun_loader import LSUN
from train.train_utils import get_transforms
from models.get_classes import get_classes
from clip import clip
from imagenet_templates import *

mix_thres = {
    'cifar-10': -0.95,
    'cifar-100': -0.65,
}

ood_dataset = {
    'cifar-10': lambda data_dir, transform: datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform),
    'cifar-100': lambda data_dir, transform: datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform),
    'cub200_ood': lambda data_dir, transform: Cub100(root=data_dir, train=False, transform=transform, id=False, mode='ood'),
    'food101_ood': lambda data_dir, transform: Food101_50(root=data_dir, split='test', id=False, transform=transform, mode='ood'),
    'stanford_cars_ood': lambda data_dir, transform: StanfordCars98(root=data_dir, split='test', id=False, transform=transform, mode='ood'),
    'oxford_iiit_pet_ood': lambda data_dir, transform: OxfordIIITPet_18(root=data_dir, split='test', id=False, transform=transform, mode='ood'),
    'dtd': lambda data_dir, transform: datasets.ImageFolder(data_dir, transform=transform),
    'svhn': lambda data_dir, transform: datasets.SVHN(data_dir, split='test', download=True, transform=transform),
    'places365': lambda data_dir, transform: datasets.ImageFolder(data_dir, transform=transform),
    'lsun': lambda data_dir, transform: LSUN(data_dir, transform=transform),
    'lsun-r': lambda data_dir, transform: datasets.ImageFolder(data_dir, transform=transform),
    'isun': lambda data_dir, transform: datasets.ImageFolder(data_dir, transform=transform),
    'imagenet20': lambda data_dir, transform: datasets.ImageFolder(os.path.join(data_dir, 'val'), 
                                transform=transform,
                                is_valid_file=lambda x: os.path.splitext(x)[1].lower() in ['.jpg', '.jpeg']
                                ),
    'imagenet10': lambda data_dir, transform: datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                transform=transform,
                                is_valid_file=lambda x: os.path.splitext(x)[1].lower() in ['.jpg', '.jpeg']
                                ),
    'imagenet100': lambda data_dir, transform: datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                transform=transform,
                                is_valid_file=lambda x: os.path.splitext(x)[1].lower() in ['.jpg', '.jpeg']
                                ),
    'imagenet1k': lambda data_dir, transform: datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                transform=transform,
                                is_valid_file=lambda x: os.path.splitext(x)[1].lower() in ['.jpg', '.jpeg']
                                )
}
concat = lambda x: np.concatenate(x, axis=0)  # 合并数组的函数
to_np = lambda x: x.data.cpu().numpy()  # 将张量转换为NumPy数组的函数


def get_id_ood_datasets(dataset_name):
    id_ood_dataset = {
        'cifar-10': ['cifar-100', 'dtd', 'places365','svhn', 'lsun', 'lsun-r', 'isun'],
        'cifar-100': ['cifar-10', 'dtd', 'places365','svhn', 'lsun', 'lsun-r', 'isun'],
        'cub200': ['cub200_ood'],
        'food101': ['food101_ood'],
        'stanford_cars': ['stanford_cars_ood'],
        'oxford_iiit_pet': ['oxford_iiit_pet_ood'],
        'imagenet20': ['imagenet10'],
        'imagenet10': ['imagenet20'],
        'imagenet100': ['dtd', 'places365','svhn', 'lsun', 'lsun-r', 'isun']
    }
    return id_ood_dataset[dataset_name]



def get_ood_scores_odin(loader, net, T = 1000, noise = 0.0014, in_dist=False, device=None):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        data = Variable(data, requires_grad = True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODIN(data, output,net, T, noise, device)
        
    return odin_score


def ODIN(inputs, outputs, model, temper, noiseMagnitude1, device=None):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    #gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
    #gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
    #gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs



def set_ood_loader(dataset_name, root_dir, batch_size, **kwargs):
    transorm = get_transforms(dataset_name, train=False)
    
    if dataset_name.startswith('imagenet'):
        data_dir = os.path.join(root_dir, dataset_name)
    elif dataset_name == 'dtd':
        data_dir = os.path.join(root_dir, 'ood_data', 'dtd', 'images')
    elif dataset_name == 'svhn':
        data_dir = os.path.join(root_dir, 'ood_data', 'svhn')
    elif dataset_name == 'lsun':
        data_dir = os.path.join(root_dir, 'ood_data', 'LSUN')
    elif dataset_name == 'places365':
        data_dir = os.path.join(root_dir, 'ood_data', 'places365')
    elif dataset_name == 'lsun-r':
        data_dir = os.path.join(root_dir, 'ood_data', 'LSUN_resize')
    elif dataset_name == 'isun':
        data_dir = os.path.join(root_dir, 'ood_data', 'iSUN')
    elif dataset_name == ['cifar-10', 'cifar-100','cub200_ood', 'food101_ood', 'stanford_cars_ood', 'oxford_iiit_pet_ood']:
        data_dir = root_dir
        
    ood_loader = DataLoader(ood_dataset[dataset_name](data_dir, transorm), batch_size=batch_size, shuffle=False, **kwargs)
    
    return ood_loader
        
    
    
def get_ood_scores(args, loader, model=None, device=None, in_dist=False):
    if model is None:
        raise ValueError("model is None")
    _score = []  # 
    _right_score = []  # 存储正确分数
    _wrong_score = []  # 存储错误分数
    list_softmax = []  # 存储softmax输出
    list_correct = []  # 存储正确性标记
    with torch.no_grad():  # 关闭梯度计算以节省内存
        for batch_idx, batch in enumerate(loader):  # 遍历数据加载器
            data, target = batch  # 获取数据和目标
            data = data.to(device)
            classes, num_classes = get_classes(args.dataset)
            
            # 提示词模版要改改，得看看怎么改
            if not args.ensemble:
                classes = [f"a photo of a {c}" for c in classes]
                text_inputs = clip.tokenize(classes).to(device) # (num_classes, 77)
                logits_voters, logits_per_image = model(image_inputs = data, text_inputs = text_inputs) # logits (batch_size, num_classes)
            else:
                logits_voters = clip_text_ens(model, data, classes, device)
                
            logits = logits_voters - model.rejection_threshold
            
            ## OOD detection
            if args.score == 'ova':  # OVA评分
                smax = to_np(logits)  # 转换logits为NumPy数组
                zeros = torch.zeros(logits.shape[0], 1).cuda()  # 创建零张量
                logits_extra = torch.cat((logits, zeros), dim=1)  # 连接logits和零张量
                ind_prob = to_np(F.softmax(logits_extra, dim=1)[:, :-1])  # 计算softmax概率
                p_max_in = np.max(ind_prob, axis=1)  # 计算最大概率
                p_ood = to_np(1 / (1 + torch.sum(torch.exp(logits), dim=1)))  # 计算OOD概率
                score_mix = np.minimum(1.0 - p_ood, p_max_in - mix_thres[args.dataset])  # 计算混合分数
                _score.append(-score_mix)  # 添加分数
                list_softmax.extend(score_mix)  # 添加softmax输出
            # if softmax:
            #     smax = to_np(F.softmax(output/ args.T, dim=1))
            # else:
            #     smax = to_np(output/ args.T)
            
            elif args.score == 'energy':
                # 计算能量分数
                #energy score is expected to be smaller for ID
                energy = -args.T * torch.logsumexp(logits / args.T, dim=1)
                _score.append(to_np(energy))
            elif args.score == 'max-logit':
                # 计算最大logit分数
                # 提出基于**最大logit值（MaxLogit）**的方法，而不是依赖softmax概率。
                max_logit = torch.max(logits, dim=1)[0]
                _score.append(to_np(-max_logit))
            elif args.score == 'msp':
                # 计算MSP分数
                # 提出**最大softmax概率（MSP）**作为检测分布外样本和错误分类的基线方法。
                msp_score = torch.max(F.softmax(logits, dim=1), dim=1)[0]
                _score.append(to_np(-msp_score))
            elif args.score == 'odin':
                odin_score = get_ood_scores_odin(loader, model, device=device)
                _score.append(-np.max(odin_score, 1))
                
            elif args.score == 'sigmoid':  # Sigmoid评分
                probs = logits.sigmoid()  # 计算sigmoid概率
                posterior_binary_prob = probs  # 保存后验概率
                probs = probs.log().cuda()  # 计算对数概率
                pred = logits.data.max(1, keepdim=True)[1]  # 计算预测
                prob, _pred = posterior_binary_prob.max(1)  # 获取最大后验概率
                prob = to_np(prob)  # 转换为NumPy数组
                output = logits  # 保留logits
                smax = to_np(posterior_binary_prob)  # 转换后验概率为NumPy数组
                _score.append(-np.max(smax, axis=1))  # 添加最大后验概率的负值
                list_softmax.extend(prob)  # 添加后验概率
                
                
            # Misclassification detection
            if in_dist:
                preds = np.argmax(smax, axis=1)  # 获取预测类别
                targets = target.numpy().squeeze()  # 获取真实标签
                right_indices = preds == targets  # 确定正确预测的索引
                wrong_indices = np.invert(right_indices)  # 确定错误预测的索引
                
                for j in range(len(preds)):  # 遍历预测结果
                    if preds[j] == target[j]:  # 如果预测正确
                        cor = 1  # 记录为正确
                    else:
                        cor = 0  # 记录为错误
                    list_correct.append(cor)  # 添加正确性标记
                    
                # 根据评分方式计算正确和错误的分数
                if args.use_xent:  # 如果使用交叉熵
                    _right_score.append(to_np((logits.mean(1) - torch.logsumexp(logits, dim=1)))[right_indices])  # 正确分类的交叉熵分数
                    _wrong_score.append(to_np((logits.mean(1) - torch.logsumexp(logits, dim=1)))[wrong_indices])  # 错误分类的交叉熵分数
                elif args.score == 'ova':
                    _right_score.append(-score_mix[right_indices])  # 正确分类的OVA分数
                    _wrong_score.append(-score_mix[wrong_indices])  # 错误分类的OVA分数
                elif args.score == 'sigmoid':
                    _right_score.append(-np.max(smax[right_indices], axis=1))  # 正确分类的Sigmoid分数
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))  # 错误分类的Sigmoid分数
            
            else:  # OOD情况下
                for j in range(len(data)):
                    cor = 0  # OOD数据标记为错误
                    list_correct.append(cor)  # 添加错误标记
 

            
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy(), list_softmax, list_correct
    else:
        return concat(_score).copy(), list_softmax, list_correct  # 仅返回OOD示例的分数

            
            
            
def clip_text_ens(model, data, classes, device):
    
    net = model.model.clip_model
    prompt_pool = openai_imagenet_template_subset[0]
    all_prompts = [template(label) for label in classes for template in prompt_pool]
    text_inputs = clip.tokenizer(all_prompts, padding=True, return_tensors="pt") # (num_classes*len(prompt_pool), 77)
    
    all_encoded_features = net.encode_text(text_inputs.to(device)).float() # (num_classes*len(prompt_pool), 512)

    all_text_features = torch.zeros(len(classes), all_encoded_features.shape[1]).to(all_encoded_features)
    for i in range(len(classes)):
        for j in range(len(prompt_pool)):
            index = i * len(prompt_pool) + j
            tmp = all_encoded_features[index] / all_encoded_features[index].norm(dim=-1, keepdim=True)
            all_text_features[i] += tmp

    all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)
    
    image_features = model.model.image_encoder(data.to(device))
    x = model.model.visual_adapter(image_features)
    ratio = model.model.ratio
    image_features = ratio * x + (1 - ratio) * image_features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logit_scale = model.model.logit_scale.exp()
    logits = logit_scale * image_features @ all_text_features.t()
    return logits


