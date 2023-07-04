import cv2
from torch.utils.data import Dataset
from pathlib import Path
import os
import numpy as np

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transforms=None, mask_suffix: str = '',if_train_aug=False,train_aug_iter=1,ratio=0.8,dev=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.transforms=transforms
        self.single_cell_mask_crop_bank=[]
        if if_train_aug:
            samples = [os.path.splitext(file)[0] for file in sorted(os.listdir(images_dir)) if not file.startswith('.')]
            if dev:
                self.ids = samples[-int(ratio * len(samples)):]
            else:
                self.ids = samples[:int(ratio*len(samples))]
            tmp=[]
            for i in range(train_aug_iter):
                tmp+=self.ids
            self.ids=tmp
        else:
            self.ids = [os.path.splitext(file)[0] for file in sorted(os.listdir(images_dir)) if not file.startswith('.')]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, pil_mask,transforms):
        tensor_img,tensor_mask=transforms(pil_img,pil_mask)
        return tensor_img,tensor_mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        mask = cv2.imread(mask_file[0].as_posix(),0) > 0
        mask = mask.astype('float32')
        img=cv2.imread(img_file[0].as_posix(),0).astype('float32')
        img=(255 * ((img - img.min()) / (img.ptp()+1e-6))).astype(np.uint8)

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        tensor_img,tensor_mask = self.preprocess(img, mask,self.transforms)

        return {
            'image': tensor_img,
            'mask': tensor_mask,
        }


