import numpy as np
import json
import os
import pickle

root_path =r'C:\code\voters\data'

def obtain_ood_oxford_iiit_pet_classes():
    loc = 'oxford-iiit-pet/ood_18_classes.pkl'
    loc = os.path.join(root_path, loc)
    if os.path.exists(loc):
        with open(loc, 'rb') as f:
            ood_classes = pickle.load(f)
    else:
        raise FileNotFoundError("File not found")
    return ood_classes

def obtain_id_oxford_iiit_pet_classes():
    loc = 'oxford-iiit-pet/id_18_classes.pkl'
    loc = os.path.join(root_path, loc)
    if os.path.exists(loc):
        with open(loc, 'rb') as f:
            selected_classes = pickle.load(f)
    else:
        raise FileNotFoundError("File not found")
    return selected_classes

def obtain_ood_stanford_cars_classes():
    loc = 'stanford_cars/ood_98_classes.pkl'
    loc = os.path.join(root_path, loc)
    if os.path.exists(loc):
        with open(loc, 'rb') as f:
            ood_classes = pickle.load(f)
    else:
        raise FileNotFoundError("File not found")
    return ood_classes

def obtain_id_stanford_cars_classes():
    loc = 'stanford_cars/id_98_classes.pkl'
    loc = os.path.join(root_path, loc)
    if os.path.exists(loc):
        with open(loc, 'rb') as f:
            selected_classes = pickle.load(f)
    else:
        raise FileNotFoundError("File not found")
    return selected_classes

def obtain_id_food101_classes():
    loc = 'food-101/id_50_classes.pkl'
    loc = os.path.join(root_path, loc)
    if os.path.exists(loc):
        with open(loc, 'rb') as f:
            selected_classes = pickle.load(f)
    else:
        raise FileNotFoundError("File not found")
    return selected_classes

def obtain_ood_food101_classes():
    loc = 'food-101/ood_50_classes.pkl'
    loc = os.path.join(root_path, loc)
    if os.path.exists(loc):
        with open(loc, 'rb') as f:
            ood_classes = pickle.load(f)
    else:
        raise FileNotFoundError("File not found")
    return ood_classes

def obtain_id_CUB100_classes():
    loc = 'CUB_200_2011/id_100_classes.pkl'
    loc = os.path.join(root_path, loc)
    if os.path.exists(loc):
        with open(loc, 'rb') as f:
            id_classes = pickle.load(f)
    else:
        raise FileNotFoundError("File not found")
    return id_classes

def obtain_ood_CUB100_classes():
    loc = 'CUB_200_2011/ood_100_classes.pkl'
    loc = os.path.join(root_path, loc)
    if os.path.exists(loc):
        with open(loc, 'rb') as f:
            ood_classes = pickle.load(f)
    else:
        raise FileNotFoundError("File not found")
    return ood_classes

def obtain_ImageNet_classes():
    loc = 'imagenet1k'
    loc = os.path.join(root_path, loc)
    with open(os.path.join(loc, 'imagenet_class_clean.npy'), 'rb') as f:
        imagenet_cls = np.load(f)
    return imagenet_cls.tolist()

def obtain_ImageNet100_classes():
    loc= 'Imagenet100'
    loc = os.path.join(root_path, loc)
    # sort by values
    with open(os.path.join(loc, 'class_list.txt')) as f:
        class_set = [line.strip() for line in f.readlines()]

    class_name_set = []
    with open(os.path.join(root_path,'imagenet1k/imagenet_class_index.json')) as file: 
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid, class_name in class_index_raw.values()}
        class_name_set = [class_index[c] for c in class_set]
    class_name_set = [x.replace('_', ' ') for x in class_name_set]

    return class_name_set


def obtain_ImageNet10_classes():

    class_dict = {"warplane": "n04552348", "sports car": "n04285008",
                  'brambling bird': 'n01530575', "Siamese cat": 'n02123597',
                  'antelope': 'n02422699', 'swiss mountain dog': 'n02107574',
                  "bull frog": "n01641577", 'garbage truck': "n03417042",
                  "horse": "n02389026", "container ship": "n03095699"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[1])}
    return list(class_dict.keys())

def obtain_ImageNet20_classes():

    class_dict = {"n04147183": "sailboat", "n02951358": "canoe", "n02782093": "balloon", "n04389033": "tank", "n03773504": "missile",
                  "n02917067": "bullet train", "n02317335": "starfish", "n01632458": "spotted salamander", "n01630670": "common newt", "n01631663": "eft",
                  "n02391049": "zebra", "n01693334": "green lizard", "n01697457": "African crocodile", "n02120079": "Arctic fox", "n02114367": "timber wolf",
                  "n02132136": "brown bear", "n03785016": "moped", "n04310018": "steam locomotive", "n04266014": "space shuttle", "n04252077": "snowmobile"}
    # sort by values
    class_dict = {k: v for k, v in sorted(
        class_dict.items(), key=lambda item: item[0])}
    return list(class_dict.values())


def get_classes(dataset):
    """
    获取指定数据集的类别。
    参数:
        dataset (string): 数据集名称。
    返回:
        list: 数据集类别列表。
    """
    
    classes = ['OOD']  # 初始化类别列表
    nums_classes = 1  # 类别数量
    # CIFAR-10 数据集类别
    
    if dataset == 'cifar-10':
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # CIFAR-100 数据集类别
    elif dataset == 'cifar-100':
        classes = [
        'beaver', 'dolphin', 'otter', 'seal', 'whale',
        'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
        'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
        'bottles', 'bowls', 'cans', 'cups', 'plates',
        'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
        'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
        'bed', 'chair', 'couch', 'table', 'wardrobe',
        'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
        'bear', 'leopard', 'lion', 'tiger', 'wolf',
        'bridge', 'castle', 'house', 'road', 'skyscraper',
        'cloud', 'forest', 'mountain', 'plain', 'sea',
        'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
        'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
        'crab', 'lobster', 'snail', 'spider', 'worm',
        'baby', 'boy', 'girl', 'man', 'woman',
        'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
        'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
        'maple', 'oak', 'palm', 'pine', 'willow',
        'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
        'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
    ]
    elif dataset == 'dtd':
        classes = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 
                   'crosshatched', 'crystalline', 'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 
                   'gauzy', 'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted', 'lacelike', 'lined', 
                   'marbled', 'matted', 'meshed', 'paisley', 'perforated', 'pitted', 'pleated', 'polka-dotted', 
                   'porous', 'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled', 'stained', 'stratified', 
                   'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']
        
    elif dataset == 'svhn':
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif dataset == 'imagenet10':
        classes = obtain_ImageNet10_classes()
    elif dataset == 'imagenet20':
        classes = obtain_ImageNet20_classes()
    elif dataset == 'imagenet100':
        classes = obtain_ImageNet100_classes()
    elif dataset == 'imagenet1k':
        classes = obtain_ImageNet_classes()
    elif dataset in ['cub200_in', 'cub200']:
        classes = obtain_id_CUB100_classes()
    elif dataset == 'cub200_ood':
        classes = obtain_ood_CUB100_classes()
    elif dataset in ['food101_in','food101']:
        classes = obtain_id_food101_classes()
    elif dataset == 'food101_ood':
        classes = obtain_ood_food101_classes()
    elif dataset in ['stanford_cars_in', 'stanford_cars']:
        classes = obtain_id_stanford_cars_classes()
    elif dataset == 'stanford_cars_ood':
        classes = obtain_ood_stanford_cars_classes()
    elif dataset in ['oxford_iiit_pet_in', 'oxford_iiit_pet']:
        classes = obtain_id_oxford_iiit_pet_classes()
    elif dataset == 'oxford_iiit_pet_ood':
        classes = obtain_ood_oxford_iiit_pet_classes()
    nums_classes = len(classes)  # 获取类别数量
    return classes, nums_classes  # 返回类别列表和数量





if __name__ == '__main__':
    classes, nums_classes = get_classes('food101')
    print(classes)
    print(nums_classes)