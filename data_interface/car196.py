import os
import pathlib
from typing import Callable, Optional, Any, Tuple
import pickle
import random

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)
        root = os.path.expanduser(root)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = self._base_folder / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
                
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.class_names_str = self.classes


    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.mode != 'all':
            # print(target)
            class_name = self.all_classes[target]
            target = self.folder_to_target[class_name]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target


    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()


class StanfordCars98(StanfordCars):
    def __init__(
        self,
        root: str = '/home/yuantongxin/whx/BCLIP2/data',
        split: str = "train",
        id: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        mode = 'all'
    ) -> None:
        super().__init__(root, split=split, transform=transform, target_transform=target_transform)
        assert mode in ['all', 'id', 'ood'], "mode must be 'all', 'id' or 'ood'"
        self.mode = mode
        
        if id:
            if mode == 'ood':
                raise ValueError("id mode cannot be ood")
        else:
            if mode == 'id':
                raise ValueError("ood mode cannot be id")
            
        subset_classes_file = os.path.join(root, 'stanford_cars', "selected_98_classes.pkl")
        if os.path.exists(subset_classes_file):
            with open(subset_classes_file, 'rb') as f:
                stored_classes = pickle.load(f)
        else:
            stored_classes = sorted(random.sample(self.classes, 98))
            with open(subset_classes_file, 'wb') as f:
                pickle.dump(stored_classes, f)

        selected_classes = stored_classes if id else [cls for cls in self.classes if cls not in stored_classes]
        self.ood_class_name_str = [cls for cls in self.classes if cls not in stored_classes] if id else stored_classes
        if mode != 'all':
            self._folder_to_target(selected_classes)

        self.selected_class_indices = [self.class_to_idx[cls] for cls in selected_classes]
        self._samples = [(image_path, target) for image_path, target in self._samples if target in self.selected_class_indices]

        self.classes = selected_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.class_names_str = self.classes
    
    def _folder_to_target(self, selected_classes):
        if self.mode == 'id':
            save_path = os.path.join(self.root, 'stanford_cars', 'id_98_classes.pkl')           
        elif self.mode == 'ood':
            save_path = os.path.join(self.root, 'stanford_cars', 'ood_98_classes.pkl')
        else:
            raise ValueError("mode must be 'id' or 'ood'")
        self.all_classes = self.classes
        self.folder_to_target = {folder: idx for idx, folder in enumerate(selected_classes)}
        classes_name =  selected_classes
        try:
            if not os.path.exists(save_path):
                with open(save_path, 'wb') as f:
                    pickle.dump(classes_name, f)
                print(f"File saved to: {save_path}")
        except Exception as e:
            print(f"Error occurred when saving file: {e}")
            
    
if __name__ == '__main__':
    dataset = StanfordCars98(root='/home/yuantongxin/whx/BCLIP2/data', split='train', id=True, mode='id')
    dataset[1000]
    import pickle
    with open('/home/yuantongxin/whx/BCLIP2/data/stanford_cars/ood_98_classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    for idx, class_name in enumerate(classes):
        if class_name == 'Volvo 240 Sedan 1993':
            print(idx)
    print(classes[96])