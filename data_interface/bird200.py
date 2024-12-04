import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import numpy as np
import pickle


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    def __init__(self, root, train=True, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self._load_metadata()

    
    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        
        # img_id      filepath                                           target  is_training_img
        # 1           001.Black_footed_Albatross/Black_Footed_...         1      1
        # 2           001.Black_footed_Albatross/Black_Footed_...         1      0
        # ...

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                         sep=' ', names=['class_id', 'target'])
        self.class_names_str = [name.split(".")[1].replace('_', ' ') for name in class_names.target]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.mode != 'all':
            folder = sample.filepath.split('/')[0]
            # print(folder)
            target = self.folder_to_target[folder]
        else:
            target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        return img, target



class Cub100(Cub2011):
    def __init__(self, root = '/home/yuantongxin/whx/BCLIP2/data', train=True, id=True, transform=None, loader=default_loader, mode = 'all'):
        self.id = id
        assert mode in ['all', 'id', 'ood'], "mode必须是'all', 'id'或'ood'"
        self.mode = mode
        self.root = root
        self._select_or_load_classes()
        super().__init__(root, train, transform, loader)
    
    def _folder_to_target(self, selected_classes):
        if self.mode == 'id':
            save_path = os.path.join(self.root, 'CUB_200_2011', 'id_100_classes.pkl')
            self.folder_to_target = {folder: idx for idx, folder in enumerate(selected_classes)}
            
        elif self.mode == 'ood':
            # 获取所有类别文件夹
            save_path = os.path.join(self.root, 'CUB_200_2011', 'ood_100_classes.pkl')
            path = os.path.join(self.root, 'CUB_200_2011', 'images')
            class_folders = sorted([d for d in os.listdir(path) 
                                    if os.path.isdir(os.path.join(path, d))])
            ood_classes = [folder for folder in class_folders if folder not in selected_classes]
            self.folder_to_target = {folder: idx for idx, folder in enumerate(ood_classes)}
        else:
            raise ValueError("mode必须是'id'或'ood'")
        
        classes_name = [name.split(".")[1].replace('_', ' ') for name in self.folder_to_target.keys()]
        try:
            if not os.path.exists(save_path):
                with open(save_path, 'wb') as f:
                    pickle.dump(classes_name, f)
                print(f"文件已保存到: {save_path}")
        except Exception as e:
            print(f"保存文件时发生错误: {e}")
            
    def _select_or_load_classes(self):
        subset_classes_file = os.path.join(self.root, 'CUB_200_2011', 'selected_100_classes.pkl')
        if os.path.exists(subset_classes_file):
            with open(subset_classes_file, 'rb') as f:
                self.selected_classes = pickle.load(f)
        else:
            all_classes = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                                      sep=' ', names=['class_id', 'target'])
            selected_class_ids = np.sort(np.random.choice(all_classes['class_id'], 100, replace=False))
            self.selected_classes = all_classes[all_classes['class_id'].isin(selected_class_ids)]['target'].tolist()
            # self.selected_classes = all_classes['target'].iloc[:100].tolist()
            with open(subset_classes_file, 'wb') as f:
                pickle.dump(self.selected_classes, f)
                
        if self.mode != 'all':
            self._folder_to_target(self.selected_classes)

    def _load_metadata(self):
        super()._load_metadata()
        all_classes = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
                              sep=' ', names=['class_id', 'target'])
    
        selected_class_ids = all_classes[all_classes['target'].isin(self.selected_classes)]['class_id'].tolist()

        if self.id: # select cub100_iid
            self.data = self.data[self.data['target'].isin(selected_class_ids)]
            self.class_names_str = [name.split(".")[1].replace('_', ' ') for name in self.selected_classes]
            
            remaining_class_ids = set(all_classes['class_id']) - set(selected_class_ids)
            remaining_class_names = all_classes[all_classes['class_id'].isin(remaining_class_ids)]['target'].tolist()
            self.ood_class_name_str = [name.split(".")[1].replace('_', ' ') for name in remaining_class_names]
        else:  # select cub100_ood
            remaining_class_ids = set(all_classes['class_id']) - set(selected_class_ids)
            self.data = self.data[self.data['target'].isin(remaining_class_ids)]
            remaining_class_names = all_classes[all_classes['class_id'].isin(remaining_class_ids)]['target'].tolist()
            self.class_names_str = [name.split(".")[1].replace('_', ' ') for name in remaining_class_names]
            


if __name__ == '__main__':
    id_dataset = Cub100(root='/home/yuantongxin/whx/BCLIP2/data', train=True, id=True, mode='id')
    ood_dataset = Cub100(root='/home/yuantongxin/whx/BCLIP2/data', train=True, id=False, mode='ood')
    import pickle
    with open('/home/yuantongxin/whx/BCLIP2/data/CUB_200_2011/id_100_classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    idx = id_dataset[1000][1]
    print(classes[idx])
    
        