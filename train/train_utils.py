import os
import torch
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet, ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from data_interface.bird200 import Cub100
from data_interface.food101 import Food101_50
from data_interface.car196 import StanfordCars98
from data_interface.pet37 import OxfordIIITPet_18


def get_transforms(dataset_name, train=True):
    """
    Get CLIP-specific transforms for different datasets
    CLIP uses a 224x224 input size and specific normalization parameters
    """
    if train:
        if dataset_name in ['cifar10', 'cifar100']:
            transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                  (0.26862954, 0.26130258, 0.27577711))
            ])
            if train:
                transform.transforms.insert(1, transforms.RandomHorizontalFlip())
        else:  # ImageNet和其他数据集
            transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                  (0.26862954, 0.26130258, 0.27577711))
            ])
            if train:
                transform.transforms.insert(1, transforms.RandomHorizontalFlip())
    else:
        if dataset_name in ['dtd', 'places365']:
            transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                  (0.26862954, 0.26130258, 0.27577711))
            ])
        elif dataset_name in ['svhn', 'lsun', 'lsun-r', 'isun']:
            transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                  (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                  (0.26862954, 0.26130258, 0.27577711))
            ])
    
    return transform

def dataset_to_dataloader(dataset, data_dir, train_batch_size, test_batch_size, **kwargs):
    transform_train = get_transforms(dataset, train=True)
    transform_test = get_transforms(dataset, train=False)
    
    if dataset == 'cifar-10':
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    elif dataset == 'cifar-100':
        train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    elif dataset == 'cub200':
        train_set = Cub100(root=data_dir, train=True, id=True, transform=transform_train, mode='id')
        test_set = Cub100(root=data_dir, train=False, id=True, transform=transform_test, mode='id')
    elif dataset == 'food101':
        train_set = Food101_50(root=data_dir, split='train', id=True, transform=transform_train, mode='id')
        test_set = Food101_50(root=data_dir, split='test', id=True, transform=transform_test, mode='id')
    elif dataset == 'stanford_cars':
        train_set = StanfordCars98(root=data_dir, split='train', id=True, transform=transform_train, mode='id')
        test_set = StanfordCars98(root=data_dir, split='test', id=True, transform=transform_test, mode='id')
    elif dataset == 'oxford_iiit_pet':
        train_set = OxfordIIITPet_18(root=data_dir, split='trainval', id=True, transform=transform_train, mode='id')
        test_set = OxfordIIITPet_18(root=data_dir, split='test', id=True, transform=transform_test, mode='id')
    elif dataset in ['imagenet20', 'imagenet10', 'imagenet100', 'imagenet1k']:
        data_dir = os.path.join(data_dir, dataset)
        train_set = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=transform_train,
            is_valid_file=lambda x: os.path.splitext(x)[1].lower() in ['.jpg', '.jpeg']
        )
        test_set = datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=transform_test,
            is_valid_file=lambda x: os.path.splitext(x)[1].lower() in ['.jpg', '.jpeg']
        )
    
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, persistent_workers=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, persistent_workers=True, **kwargs)
    return train_loader, test_loader