import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

import glob
import cv2
import torchio as tio
from torch.utils.data import DataLoader

PREPROCESSING_TRANSORMS_CT = tio.Compose([
    tio.Clamp(out_min=-250, out_max=800),
    tio.RescaleIntensity(in_min_max=(-250, 800),
                         out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(64, 64, 64))
])
PREPROCESSING_TRANSORMS_MRI = tio.Compose([
    tio.Clamp(out_min=0, out_max=1000),
    tio.RescaleIntensity(in_min_max=(0, 1000),
                         out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(64, 64, 64))
])

PREPROCESSING_MASK_TRANSORMS = tio.Compose([
    tio.CropOrPad(target_shape=(64, 64, 64))
])

class MMWHS_Dataset(Dataset):
    def __init__(self, root_dir='', data_type='', mode = ''):
        self.root_dir = root_dir
        self.data_type = data_type
        self.mode = mode
        self.file_names = self.get_file_names()
        self.preprocessing_img_ct = PREPROCESSING_TRANSORMS_CT
        self.preprocessing_img_mri = PREPROCESSING_TRANSORMS_MRI
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSORMS
    
    @staticmethod
    def create_mask(shape):
        return torch.zeros(shape, dtype=torch.uint8)

    @staticmethod 
    def project_to_2d(mask):
        projection = torch.max(mask, dim=0)[0]
        return projection.numpy()

    @staticmethod
    def min_enclosing_circle(projection):

        points = np.column_stack(np.where(projection > 0))
        points = points.astype(np.float32)
        print(points.shape)       
        (x, y), radius = cv2.minEnclosingCircle(points.astype(np.float32))
        center = (int(x), int(y))
        radius = int(radius)
        return center, radius

    @staticmethod
    def create_circle_mask_2d(shape, center, radius):
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, thickness=-1)
        return mask

    @staticmethod
    def apply_circle_mask_to_3d(mask, circle_mask_2d):
        for i in range(mask.shape[0]):
            mask[i] = torch.from_numpy(circle_mask_2d)
        return mask
    
    def train_transform(self, image, label, sdf, p):
        TRAIN_TRANSFORMS = tio.Compose([
            tio.RandomFlip(axes=(1), flip_probability=p),
        ])
        image = TRAIN_TRANSFORMS(image)
        label = TRAIN_TRANSFORMS(label)
        sdf = TRAIN_TRANSFORMS(sdf)
        return image, label, sdf

    def get_file_names(self):
        all_img_names = glob.glob(os.path.join(self.root_dir, './*image.nii.gz'), recursive=True)
        return all_img_names

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        mask_path = img_path.replace("image.nii.gz", "label.nii.gz")
        sdf_path = img_path.replace("image.nii.gz", "sdf.nii.gz")

        img = tio.ScalarImage(img_path)
        mask = tio.LabelMap(mask_path) 
        sdf = tio.ScalarImage(sdf_path) 
        name = img_path.split('/')[-1]
        name = name.split('.nii')[0]
        
        if self.data_type.upper() == 'CT':
            img = self.preprocessing_img_ct(img)
        elif self.data_type.upper() == 'MRI':
            img = self.preprocessing_img_mri(img)
        else:
            raise ValueError("Wrong Data Type!")
        mask = self.preprocessing_mask(mask)

        p = np.random.choice([0, 1])
        if self.mode == 'train':
            img, mask, sdf = self.train_transform(img, mask, sdf, p)
        
        affine = img.affine
        mask = mask.data
        img = img.data
        sdf = sdf.data

        label_1 = (mask == 1).float()
        label_2 = (mask == 2).float()
        label_3 = (mask == 3).float()
        label_4 = (mask == 4).float()
        label_5 = (mask == 5).float()
        label = torch.cat((label_1, label_2, label_3, label_4, label_5), dim=0)
        label_sdf = sdf

        return {
            'name': name,
            'img': img,
            'mask_sdf': label_sdf,
            'mask': label,
            'affine': affine
        }

def get_MMWHS_dataloader(root_dir, data_type, mode, batch_size=1, drop_last=False):
    dataset = MMWHS_Dataset(root_dir=root_dir, data_type=data_type, mode=mode)
    if mode == 'train':
        shuffle = True
        return dataset
    elif mode == 'test':
        shuffle = False
    else:
        raise ValueError('No Such Mode')
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=20, pin_memory=True, drop_last=drop_last
    )
    return loader

