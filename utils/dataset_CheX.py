import os
import random
import h5py
import numpy as np
import torch
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator_CheX(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample
    
class ResizeGenerator_CheX(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class CheX_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, nclass=9, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.nclass = nclass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            img_data_path = os.path.join(self.data_dir, 'images', 'image_' + slice_name + '.jpg')
            mask_data_path = os.path.join(self.data_dir, 'masks', 'mask_' + slice_name + '.png')
            assert os.path.exists(img_data_path), f"Image file does not exist: {img_data_path}"
            assert os.path.exists(mask_data_path), f"Mask file does not exist: {mask_data_path}"
            image = np.asarray(cv2.imread(img_data_path, cv2.IMREAD_GRAYSCALE))
            label = np.asarray(cv2.imread(mask_data_path, 0))
            label[label <= 0] = 0
            label[label > 0] = 1
            # print(image.shape, label.shape)
            # image = cv2.resize(image, (self.image_size, self.image_size))
            #print(image.shape)
            #image = np.reshape(image, (512, 512))
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            #label = np.reshape(label, (512, 512))
            
            
        else:
            slice_name = self.sample_list[idx].strip('\n')
            img_data_path = os.path.join(self.data_dir, 'images', 'image_' + slice_name + '.jpg')
            mask_data_path = os.path.join(self.data_dir, 'masks', 'mask_' + slice_name + '.png')
            assert os.path.exists(img_data_path), f"Image file does not exist: {img_data_path}"
            assert os.path.exists(mask_data_path), f"Mask file does not exist: {mask_data_path}"
            image = np.asarray(cv2.imread(img_data_path, cv2.IMREAD_GRAYSCALE))
            label = np.asarray(cv2.imread(mask_data_path, 0))
            label[label <= 0] = 0
            label[label > 0] = 1
            #image = np.reshape(image, (image.shape[2], 512, 512))
            #label = np.reshape(label, (label.shape[2], 512, 512))
            #label[label==5]= 0
            #label[label==9]= 0
            #label[label==10]= 0
            #label[label==12]= 0
            #label[label==13]= 0
            #label[label==11]= 5
            
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
