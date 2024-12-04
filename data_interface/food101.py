from pathlib import Path
import json
from typing import Any, Tuple, Callable, Optional
import torch
import PIL.Image
import os
import pickle
import random

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset

class Food101(VisionDataset):
    """`The Food-101 Data Set <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>`_.

    The Food-101 is a challenging data set of 101 food categories, with 101'000 images.
    For each class, 250 manually reviewed test images are provided as well as 750 training images.
    On purpose, the training images were not cleaned, and thus still contain some amount of noise.
    This comes mostly in the form of intense colors and sometimes wrong labels. All images were
    rescaled to have a maximum side length of 512 pixels.


    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = Path(self.root) / "food-101"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"
        self.class_names_str = ['Apple pie', 'Baby back ribs', 'Baklava', 'Beef carpaccio', 'Beef tartare', 'Beet salad', 'Beignets', 'Bibimbap', 'Bread pudding', 'Breakfast burrito', 'Bruschetta', 'Caesar salad', 'Cannoli', 'Caprese salad', 'Carrot cake', 'Ceviche', 'Cheesecake', 'Cheese plate', 'Chicken curry', 'Chicken quesadilla', 'Chicken wings', 'Chocolate cake', 'Chocolate mousse', 'Churros', 'Clam chowder', 'Club sandwich', 'Crab cakes', 'Creme brulee', 'Croque madame', 'Cup cakes', 'Deviled eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs benedict', 'Escargots', 'Falafel', 'Filet mignon', 'Fish and chips', 'Foie gras', 'French fries', 'French onion soup', 'French toast', 'Fried calamari', 'Fried rice', 'Frozen yogurt', 'Garlic bread', 'Gnocchi', 'Greek salad', 'Grilled cheese sandwich', 'Grilled salmon', 'Guacamole', 'Gyoza', 'Hamburger', 'Hot and sour soup', 'Hot dog', 'Huevos rancheros', 'Hummus', 'Ice cream', 'Lasagna', 'Lobster bisque', 'Lobster roll sandwich', 'Macaroni and cheese', 'Macarons', 'Miso soup', 'Mussels', 'Nachos', 'Omelette', 'Onion rings', 'Oysters', 'Pad thai', 'Paella', 'Pancakes', 'Panna cotta', 'Peking duck', 'Pho', 'Pizza', 'Pork chop', 'Poutine', 'Prime rib', 'Pulled pork sandwich', 'Ramen', 'Ravioli', 'Red velvet cake', 'Risotto', 'Samosa', 'Sashimi', 'Scallops', 'Seaweed salad', 'Shrimp and grits', 'Spaghetti bolognese', 'Spaghetti carbonara', 'Spring rolls', 'Steak', 'Strawberry shortcake', 'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna tartare', 'Waffles']

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())

        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._images_folder.joinpath(*f"{im_rel_path}.jpg".split("/")) for im_rel_path in im_rel_paths
            ]
        # self.classes = ['a', 'b', ...]
        # self._labels = [0,0,...,1,1,...,2,2,...]
        # self._image_files = [path1, path2,...,path3, path4,...,path5, path6,...]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.mode != 'all':
            folder = image_file.parts[-2]
            # print(folder)
            label = self.folder_to_target[folder]

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return all(folder.exists() and folder.is_dir() for folder in (self._meta_folder, self._images_folder))

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._URL, download_root=self.root, md5=self._MD5)


class Food101_50(Food101):
    def __init__(
        self,
        root: str = '/home/yuantongxin/whx/BCLIP2/data',
        split: str = "train",
        id: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        mode: str = 'all'
        
    ) -> None:        
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        assert mode in ['all', 'id', 'ood'], "mode must be 'all', 'id' or 'ood'"
        
        if id:
            if mode == 'ood':
                raise ValueError("id mode cannot be ood")
        else:
            if mode == 'id':
                raise ValueError("ood mode cannot be id")
            
        self.mode = mode
        selected_classes_file = os.path.join(root, 'food-101', 'selected_50_classes.pkl')
        if os.path.exists(selected_classes_file):
            with open(selected_classes_file, 'rb') as f:
                selected_classes = pickle.load(f)
        else:
            selected_classes = sorted(random.sample(self.classes, 50))
            with open(selected_classes_file, 'wb') as f:
                pickle.dump(selected_classes, f)

        selected_classes = selected_classes if id else [cls for cls in self.classes if cls not in selected_classes]
        selected_ood_classes = [cls for cls in self.classes if cls not in selected_classes] if id else selected_classes
        
        if mode != 'all':
            self._folder_to_target(selected_classes)

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(selected_classes)}

        selected_image_files = []
        selected_labels = []
        for idx, (image_file, label) in enumerate(zip(self._image_files, self._labels)):
            if self.classes[label] in selected_classes:
                selected_image_files.append(image_file)
                selected_labels.append(label)
                
        self._image_files = selected_image_files
        self._labels = selected_labels
        self.classes = selected_classes
        self.class_names_str = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls in selected_classes
        ]
        self.ood_class_name_str = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls in selected_ood_classes
        ]
        
    def _folder_to_target(self, selected_classes):
        if self.mode == 'id':
            save_path = os.path.join(self.root, 'food-101', 'id_50_classes.pkl')
 
            
        elif self.mode == 'ood':
            save_path = os.path.join(self.root, 'food-101', 'ood_50_classes.pkl')
        else:
            raise ValueError("mode must be 'id' or 'ood'")
        self.folder_to_target = {folder: idx for idx, folder in enumerate(selected_classes)}
        classes_name =  [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls in selected_classes
        ]
        try:
            if not os.path.exists(save_path):
                with open(save_path, 'wb') as f:
                    pickle.dump(classes_name, f)
                print(f"File saved to: {save_path}")
        except Exception as e:
            print(f"Error occurred when saving file: {e}")


def examine_count(counter, name = "train"):
    print(f"in the {name} set")
    for label in counter:
        print(label, counter[label])

if __name__ == "__main__":

    # label_names = []
    # with open('debug/food101_labels.txt') as f:
    #     for name in f:
    #         label_names.append(name.strip())
    # print(label_names)

    # train_set = Food101(root = "/nobackup/dataset_myf", split = "train", download = True)
    # test_set = Food101(root = "/nobackup/dataset_myf", split = "test")
    # print(f"train set len {len(train_set)}")
    # print(f"test set len {len(test_set)}")
    # from collections import Counter
    # train_label_count = Counter(train_set._labels)
    # test_label_count = Counter(test_set._labels)

    # kwargs = {'num_workers': 4, 'pin_memory': True}
    # train_loader = torch.utils.data.DataLoader(train_set ,
    #                 batch_size=16, shuffle=True, **kwargs)
    # val_loader = torch.utils.data.DataLoader(test_set,
    #                 batch_size=16, shuffle=False, **kwargs)
    test = Food101_50(root = "/home/yuantongxin/whx/BCLIP2/data", split = "test", id = False, mode = 'ood')
    with open("/home/yuantongxin/whx/BCLIP2/data/food-101/ood_50_classes.pkl", 'rb') as f:
        classes = pickle.load(f)
    idx = test[1000][1]
    print(classes[idx])