import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
from common_utils.image import imread

class ImageNoiseSet(Dataset):
    """
    A dataset comprised of crops of a single image or several images.
    """
    def __init__(self, image_path, dataset_size=6*2000,num_images = 50,use_encoder = False,noise_dim=60):
        """
        Args:
            image_path (os.path): The image path to generate image and noise from.
                                  Can be of shape (C,H,W) or (B,C,H,W) in case of several images.
            dataset_size (int): The amount of images in a single epoch of training. For training datasets,
                                this should be a high number to avoid overhead from pytorch_lightning.
        """
        self.dataset_size = dataset_size

        transform_list = []
        transform_list += [
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)  # make [3,256,256] --> [1,3,256,256]
        ]

        self.transform = transforms.Compose(transform_list)
        self.img = []
        self.noise_top_train = []
        random_seed = 3047
        torch.manual_seed(random_seed)
        if not use_encoder:
        # 获取image_path下所有的PNG文件
            png_files = [f for f in os.listdir(image_path) if f.endswith('.png')] # in an order of 001.png, 002.png...018.png
        else:
            png_files = [f for f in os.listdir(image_path) if f.endswith('.enc')]
        # 按照文件名进行排序
        png_files.sort() # in an order of 001.png, 002.png...018.png
        assert num_images<=len(png_files),'dataset is too small!'
        png_files = png_files[:num_images]

        self.num_image = len(png_files)
        if not use_encoder:
            for imgfile in png_files:
                cur_img = imread(os.path.join(image_path, imgfile))
                cur_img_trans = self.transform(cur_img)
                self.img.append(cur_img_trans[0])
                cur_noise = torch.randn((noise_dim))
                self.noise_top_train.append(cur_noise)
        else:
            for encfile in png_files:
                cur_enc = torch.load(os.path.join(image_path, encfile))
                self.img.append(cur_enc.detach()*0.07)
                cur_noise = torch.randn((noise_dim))
                self.noise_top_train.append(cur_noise)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        # If the training is multi-image, choose one of them to get the crop from
        return {'IMG': self.img[item%self.num_image],
                'TOP_TRAIN_NOISE': self.noise_top_train[item%self.num_image],
                'FRAME':item%self.num_image
                }
        # return {'IMG': self.img[item%self.num_image]}
    def get_img(self, idx):
        # If the training is multi-image, choose one of them to get the crop from
        return {'IMG': self.img[idx],
                'TOP_TRAIN_NOISE': self.noise_top_train[idx],
                'FRAME':idx
                }

# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        if_encoder: bool = False,
        num_images: int = 1000,
        dataset_size: int = 1000,
        noise_dim: int = 60,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.if_encoder = if_encoder
        self.num_images = num_images
        self.dataset_size = dataset_size
        self.noise_dim = noise_dim

    def setup(self, stage: Optional[str] = None) -> None:
        #       =========================  FFHQ Dataset  =========================
        self.train_dataset = ImageNoiseSet(image_path=self.data_dir,
                                           dataset_size=self.dataset_size,
                                           num_images= self.num_images,
                                           use_encoder=self.if_encoder,
                                           noise_dim=self.noise_dim)
     



#       ===============================================================
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        # train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                       transforms.CenterCrop(148),
        #                                       transforms.Resize(self.patch_size),
        #                                       transforms.ToTensor(),])
        
        # val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                     transforms.CenterCrop(148),
        #                                     transforms.Resize(self.patch_size),
        #                                     transforms.ToTensor(),])
        
        # self.train_dataset = MyCelebA(
        #     self.data_dir,
        #     split='train',
        #     transform=train_transforms,
        #     download=False,
        # )
        
        # # Replace CelebA with your dataset
        # self.val_dataset = MyCelebA(
        #     self.data_dir,
        #     split='test',
        #     transform=val_transforms,
        #     download=False,
        # )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    # def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.val_batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=self.pin_memory,
    #     )
    
    # def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=144,
    #         num_workers=self.num_workers,
    #         shuffle=True,
    #         pin_memory=self.pin_memory,
    #     )
     