#COD10k

"""
Code from SimAda dataset.py file https://github.com/zongzi13545329/SimAda/blob/main/dataset.py
Some changes to play with the code
"""

"""
train and test dataset
"""
import os
import sys
import pickle
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms


from PIL import Image, ImageFilter, ImageDraw

#from utils import random_click

#from monai.transforms import LoadImaged, Randomizable,LoadImage

def random_click(mask, point_labels = 1, inout = 1):
    indices = np.argwhere(mask == inout)
    return indices[np.random.randint(len(indices))][::-1].copy()

        


class COD10K(Dataset):
    #####only have test dataset
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click'):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size

        ######################################### dataset path
        assert self.mode == 'Training' or self.mode == 'Testing' or self.mode == 'Validation' \
        f'Mode argument should be one of [Training, Testing, Validation], received {self.mode} instead'
        if self.mode == 'Training':
            image_root = os.path.join(self.data_path + '/COD10K/Train/Image/')
            gt_root = os.path.join(self.data_path + '/COD10K/Train/GT_Object/')
        elif self.mode == 'Testing':
            image_root = os.path.join(self.data_path + '/COD10K/Test/Image/')
            gt_root = os.path.join(self.data_path + '/COD10K/Test/GT_Object/')

        
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        #self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize) # resize the mask to image size for some
                                  # calculations later. The final size for the
                                  # mask (gt) will be out_size.

        #returns a random point on the object in the image as prompt
        # gets the point from the mask (ground truth)
        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
          # rng : random number generator
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)

        name=self.images[index].split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample

    # Checks if images and labels are compatible (same size). Returns list of
    # compatible image paths.
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    # convert image to RGB
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    # load a binary image file (outputs/GT), convert it to grayscale, and
    # return the grayscale image
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    ## I Removed a function called resize(self,img,gt)



    def __len__(self):
        return self.size


class CAMO(Dataset):
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click'):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size

        ######################################### dataset path
        assert self.mode == 'Training' or self.mode == 'Testing' or self.mode == 'Validation' \
        f'Mode argument should be one of [Training, Testing, Validation], received {self.mode} instead'
        if self.mode == 'Training':
            image_root = os.path.join(self.data_path + '/CAMO/Train/Image/')
            gt_root = os.path.join(self.data_path + '/CAMO/Train/GT/')
        elif self.mode == 'Testing':
            image_root = os.path.join(self.data_path + '/CAMO/Test/Image/')
            gt_root = os.path.join(self.data_path + '/CAMO/Test/GT/')

        
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        #self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize) # resize the mask to image size for some
                                  # calculations later. The final size for the
                                  # mask (gt) will be out_size.

        #returns a random point on the object in the image as prompt
        # gets the point from the mask (ground truth)
        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)

        if self.transform:
          # rng : random number generator
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)

        name=self.images[index].split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample

    # Checks if images and labels are compatible (same size). Returns list of
    # compatible image paths.
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    # convert image to RGB
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    # load a binary image file (outputs/GT), convert it to grayscale, and
    # return the grayscale image
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    ## I Removed a function called resize(self,img,gt)



    def __len__(self):
        return self.size



class SBU(Dataset):
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click'):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size

        ######################################### dataset path
        assert self.mode == 'Training' or self.mode == 'Testing' or self.mode == 'Validation' \
        f'Mode argument should be one of [Training, Testing, Validation], received {self.mode} instead'
        if self.mode == 'Training':
            image_root = os.path.join(self.data_path + '/SBU/Train/Image/')
            gt_root = os.path.join(self.data_path + '/SBU/Train/Labels/')
        elif self.mode == 'Testing':
            image_root = os.path.join(self.data_path + '/SBU/Test/Image/')
            gt_root = os.path.join(self.data_path + '/SBU/Test/Labels/')

        
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        #self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1

        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize) # resize the mask to image size for some
                                  # calculations later. The final size for the
                                  # mask (gt) will be out_size.

        #returns a random point on the object in the image as prompt
        # gets the point from the mask (ground truth)
        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)



        if self.transform:
          # rng : random number generator
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)

        name=self.images[index].split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample

    # Checks if images and labels are compatible (same size). Returns list of
    # compatible image paths.
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    # convert image to RGB
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    # load a binary image file (outputs/GT), convert it to grayscale, and
    # return the grayscale image
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    ## I Removed a function called resize(self,img,gt)



    def __len__(self):
        return self.size
        
        
        
class Crops(Dataset):
    #####only have test dataset
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click'):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size
        
        ######################################### dataset path
        assert self.mode == 'Training' or self.mode == 'Testing' or self.mode == 'Validation' \
        f'Mode argument should be one of [Training, Testing, Validation], received {self.mode} instead'
        if self.mode == 'Training':
            image_root = os.path.join(self.data_path + '/Crops/All/Train/Image/')
            gt_root = os.path.join(self.data_path + '/Crops/All/Train/Labels_2/')
        elif self.mode == 'Testing':
            image_root = os.path.join(self.data_path + '/Crops/All/Test/Image/')
            gt_root = os.path.join(self.data_path + '/Crops/All/Test/Labels_2/')
        
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') 
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        #self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1


        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize) # resize the mask to image size for some
                                  # calculations later. The final size for the
                                  # mask (gt) will be out_size.

        #returns a random point on the object in the image as prompt
        # gets the point from the mask (ground truth)
        if self.prompt == 'click':
            pt = random_click(np.array(mask)/255, point_label, inout)
            
        
        

        if self.transform:
          # rng : random number generator
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)

        

        name=self.images[index].split('/')[-1]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample


    # Checks if images and labels are compatible (same size). Returns list of
    # compatible image paths.
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    # convert image to RGB
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    # load a binary image file (outputs/GT), convert it to grayscale, and
    # return the grayscale image
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    ## I Removed a function called resize(self,img,gt)



    def __len__(self):
        return self.size



class Crops_Text(Dataset):
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click'):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size
        
        ######################################### dataset path
        assert self.mode == 'Training' or self.mode == 'Testing' or self.mode == 'Validation' \
        f'Mode argument should be one of [Training, Testing, Validation], received {self.mode} instead'
        if self.mode == 'Training':
            image_root = os.path.join(self.data_path + '/Crops/All/Train/Image/')
            gt_root = os.path.join(self.data_path + '/Crops/All/Train/Labels_2/')
        elif self.mode == 'Testing':
            image_root = os.path.join(self.data_path + '/Crops/All/Test/Image/')
            gt_root = os.path.join(self.data_path + '/Crops/All/Test/Labels_2/')
        
        
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') 
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        #self.filter_files()
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        #prompt个数
        inout = 1
        point_label = 1


        image = self.rgb_loader(self.images[index])
        text_p = self.images[index].split('/')[-1].split('_')[0]
        gt = self.binary_loader(self.gts[index])
        newsize = (self.img_size, self.img_size)
        mask = gt.resize(newsize) # resize the mask to image size for some
                                  # calculations later. The final size for the
                                  # mask (gt) will be out_size.

        #returns a random point on the object in the image as prompt
        # gets the point from the mask (ground truth)
        if self.prompt == 'click':
            pt = random_click(np.array(mask)/255, point_label, inout)
            
        
        

        if self.transform:
          # rng : random number generator
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)

        if self.transform_msk:
            mask = self.transform_msk(mask)

        

        name=self.images[index].split('/')[-1]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'text':text_p,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict': image_meta_dict,
        }

        return sample

    # Checks if images and labels are compatible (same size). Returns list of
    # compatible image paths.
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    # convert image to RGB
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    # load a binary image file (outputs/GT), convert it to grayscale, and
    # return the grayscale image
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    ## I Removed a function called resize(self,img,gt)



    def __len__(self):
        return self.size

class Crops_Text_Unsupervised(Dataset):
    def __init__(self,  out_size, data_path , image_size, mode = 'Training', prompt = 'click'):
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = image_size
        self.out_size = out_size
        
        ######################################### dataset path
        if self.mode == 'Training':
            image_root = os.path.join(self.data_path + '/Crops/Unlabeled_filtered/All/')
           
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') 
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        #prompt        
        image = self.rgb_loader(self.images[index])
        text_p = self.images[index].split('/')[-1].split('_')[0]
        newsize = (self.img_size, self.img_size)


        if self.transform:
          # rng : random number generator
            state = torch.get_rng_state()
            img = self.transform(image)
            torch.set_rng_state(state)


        name=self.images[index].split('/')[-1]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'text':text_p,
            'image_meta_dict': image_meta_dict,
        }

        return sample


    # convert image to RGB
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')



    def __len__(self):
        return self.size


class DynamicCOCODataset(Dataset):
    def __init__(self, image_dir, mask_dir, category_mapping, include_categories=None, img_size=1024, out_size=256, file_list=None):
        """
        Args:
            image_dir (str): Path to the directory containing COCO images.
            mask_dir (str): Path to the directory containing ground truth masks.
            category_mapping (dict): Dictionary mapping category names to category IDs.
            include_categories (list, optional): List of category names to include in the dataset. Defaults to None (include all).
            img_size (int, optional): Size to which input images are resized. Default is 1024.
            out_size (int, optional): Size to which masks are resized. Default is 256.
            file_list (str, optional): Path to a text file containing mask filenames to include. Defaults to None (use all matching files).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.out_size = out_size
        self.category_mapping = {v: k for k, v in category_mapping.items()}  # Reverse mapping: ID -> Name
        self.transform_img = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.transform_msk = transforms.Compose([
        transforms.Resize((self.out_size, self.out_size), 
                          interpolation=transforms.InterpolationMode.NEAREST),  # Prevent soft labels 
                          transforms.ToTensor()
                          ])

        if file_list:
            # Load file list if provided
            with open(file_list, "r") as f:
                self.mask_files = f.read().splitlines()
        else:
            # Determine which category IDs to include
            if include_categories:
                self.include_category_ids = [category_mapping[cat] for cat in include_categories if cat in category_mapping]
            else:
                self.include_category_ids = list(category_mapping.values())  # Include all categories

            # Collect all mask file names that match the included categories
            self.mask_files = [
                f for f in os.listdir(mask_dir) if f.endswith('.png') and int(f.split('_')[1].split('.')[0]) in self.include_category_ids
            ]

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        inout = 1
        point_label = 1
        
        # Get mask file name
        mask_file = self.mask_files[idx]

        # Extract image index and category ID from the file name (e.g., "12345_1.png")
        try:
            img_index, cat_index = mask_file.split("_")
            cat_index = int(cat_index.split(".")[0])
        except ValueError:
            raise ValueError(f"Invalid mask file name format: {mask_file}")

        # Get the category name (prompt) from the mapping
        prompt = self.category_mapping.get(cat_index, "Unknown")

        # Load the image
        img_path = os.path.join(self.image_dir, f"{img_index}.jpg")
        image = Image.open(img_path).convert("RGB")

        # Load the mask
        mask_path = os.path.join(self.mask_dir, mask_file)
        mask = Image.open(mask_path).convert('L')
        newsize = (self.img_size, self.img_size)
        mask_resized = mask.resize(newsize)
        #mask = np.array(mask, dtype=np.uint8)  # Ensure mask is in numpy array format

        # produce the point prompt
        pt = random_click(np.array(mask_resized)/255, point_label, inout)
        mask = np.array(mask)
        
        # Apply transformations
        image = self.transform_img(image)
        mask = self.transform_msk(Image.fromarray(mask))  # Convert mask back to PIL before transforming
        return {
            'image': image,
            'label': mask,
            'text': prompt,
            'p_label':point_label,
            'pt':pt
        }


from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import json
import os

class COCOBinaryMaskDataset(Dataset):
    def __init__(self, samples_json, img_size=512, out_size=128):
        with open(samples_json, 'r') as f:
            self.samples = json.load(f)

        self.img_size = img_size
        self.out_size = out_size

        # Define the image and mask transformations
        self.transform_img = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.transform_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        mask_path = sample['mask_path']
        class_id = sample['class_id']
        class_name = sample['class_name']

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        # Generate binary mask
        binary_mask = (mask == class_id).astype(np.uint8)

        # Apply transformations
        image = self.transform_img(image)
        binary_mask = Image.fromarray(binary_mask)
        binary_mask = self.transform_msk(binary_mask)  # [1, H, W]

        # Ensure the binary mask has the correct shape and type
        binary_mask = (binary_mask > 0).float()  # make sure values are 0 and 1, and convert to float

        return {
            'image': image,
            'label': binary_mask,  # float tensor with values 0 or 1
            'text': class_name,
            'fname': os.path.basename(img_path),
            'image_path': img_path
        }



class COCOBinaryMaskDataset_wTextEmb(Dataset):
    def __init__(
        self,
        samples_json: str,
        img_size: int = 512,
        out_size: int = 128,
        emb_dir: str | None = None,        # <-- NEW
    ):
        # --------------- samples -----------------
        with open(samples_json, "r") as f:
            self.samples = json.load(f)

        self.img_size = img_size
        self.out_size = out_size

        # --------------- transforms --------------
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        self.transform_msk = transforms.Compose([
            transforms.Resize((out_size, out_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # --------------- text embeddings ----------
        self.text_embs: dict[int, torch.Tensor] = {}
        if emb_dir is not None:
            emb_dir = Path(emb_dir)
            for file in emb_dir.glob("*.pt"):
                # Expect filenames like "3_car.pt" → class_id = 3
                class_id = int(file.stem.split("_")[0])
                self.text_embs[class_id] = torch.load(file, map_location="cpu")
            if len(self.text_embs) == 0:
                raise RuntimeError(f"No .pt files found in {emb_dir}")
        else:
            raise ValueError("emb_dir must be provided so that text embeddings "
                             "can be attached to each sample.")
        
        self.class_id_list = sorted(self.text_embs.keys())



    # ---------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        img_path   = sample["image_path"]
        mask_path  = sample["mask_path"]
        class_id   = sample["class_id"]
        class_name = sample["class_name"]

        # ----------- image & mask ---------------
        image = Image.open(img_path).convert("RGB")
        mask  = np.array(Image.open(mask_path))

        binary_mask = (mask == class_id).astype(np.uint8)
        if binary_mask.max() != 1:
            print('faulty binary mask!')

        image       = self.transform_img(image)
        binary_mask = self.transform_msk(Image.fromarray(binary_mask))
        binary_mask = (binary_mask > 0).float()   # ensure {0,1} & float32

        # ----------- text embedding -------------
        try:
            text_emb = self.text_embs[class_id]          # (512,)
        except KeyError:
            raise KeyError(f"No embedding found for class_id {class_id}; "
                           "make sure matching .pt exists in emb_dir")

        # Pick a random *other* class embedding
        if len(self.class_id_list) > 1:
            # exclude current class_id
            pool = [cid for cid in self.class_id_list if cid != class_id]
            other_id = np.random.choice(pool)
            other_text_emb = self.text_embs[other_id]
        else:
            # degenerate case: only one class in emb_dir
            other_text_emb = text_emb  # fallback


        return {
            "image": image,                 # [3, H, W]
            "label": binary_mask,           # [1, H', W']
            "text":  class_name,            # str (optional)
            "class_id": class_id,           # NEW: int
            "text_emb": text_emb,           # [D]
            "other_text_emb": other_text_emb,  # NEW: [D]
            "fname": os.path.basename(img_path),
        }

class EfficientCOCOBinaryMaskDataset(Dataset):
    def __init__(self, preprocessed_path, img_size=1024, out_size=256, use_cache=True):
        with open(preprocessed_path, 'rb') as f:
            data = pickle.load(f)
        self.image_index = data['image_index']
        self.class_index_map = data['class_index_map']

        self.img_size = img_size
        self.out_size = out_size
        self.use_cache = use_cache
        self.image_cache = {} if use_cache else None

        self.transform_img = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    def __len__(self):
            return len(self.class_index_map)

    def __getitem__(self, idx):
        img_path, class_idx = self.class_index_map[idx]
        mask_path, class_id, class_name = self.image_index[img_path][class_idx]

        # Load image (once if cached)
        if self.use_cache and img_path in self.image_cache:
            image = self.image_cache[img_path]
        else:
            image = Image.open(img_path).convert("RGB")
            image = self.transform_img(image)
            if self.use_cache:
                self.image_cache[img_path] = image

        # Load and process mask
        mask = np.array(Image.open(mask_path))
        binary_mask = (mask == class_id).astype(np.uint8)  # [H, W]
        binary_mask = torch.from_numpy(binary_mask).unsqueeze(0).float()
        binary_mask = torch.nn.functional.interpolate(binary_mask.unsqueeze(0), size=(self.out_size, self.out_size), mode="nearest").squeeze(0)

        return {
            'image': image,
            'label': binary_mask,
            'text': class_name,
            'fname': os.path.basename(img_path)
        }


class COCOBinaryMaskDataset_wAUG(Dataset):
    def __init__(self, samples_json, img_size=512, out_size=128):
        with open(samples_json, 'r') as f:
            self.samples = json.load(f)

        self.img_size = img_size
        self.out_size = out_size

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        mask_path = sample['mask_path']
        class_id = sample['class_id']
        class_name = sample['class_name']

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Generate binary mask
        mask = np.array(mask)
        binary_mask = (mask == class_id).astype(np.uint8)
        binary_mask = Image.fromarray(binary_mask * 255)

        # Resize to img_size
        image = F.resize(image, [self.img_size, self.img_size])
        binary_mask = F.resize(binary_mask, [self.out_size, self.out_size], interpolation=Image.NEAREST)

        # --- Begin Augmentation Block ---
        # Random horizontal flip
        if random.random() < 0.5:
            image = F.hflip(image)
            binary_mask = F.hflip(binary_mask)

        # Random brightness / contrast
        if random.random() < 0.2:
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            image = F.adjust_brightness(image, brightness_factor)
            image = F.adjust_contrast(image, contrast_factor)

        # Random color jitter
        if random.random() < 0.2:
            color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
            image = color_jitter(image)

        # Random affine
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            translate = (random.uniform(-0.05, 0.05) * self.img_size,
                         random.uniform(-0.05, 0.05) * self.img_size)
            scale = random.uniform(0.9, 1.1)
            shear = 0.0
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
            binary_mask = F.affine(binary_mask, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=Image.NEAREST)

        # Resize to output size
        #image = F.resize(image, [self.img_size, self.img_size])
        #binary_mask = F.resize(binary_mask, [self.out_size, self.out_size], interpolation=Image.NEAREST)
        # --- End Augmentation Block ---

        # Normalize and convert to tensor
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)

        binary_mask = F.to_tensor(binary_mask)
        binary_mask = (binary_mask > 0).float()  # Ensure binary mask [0, 1]

        return {
            'image': image,
            'label': binary_mask,
            'text': class_name,
            'fname': os.path.basename(img_path)
        }


class COCOBinaryMaskDataset_wNewAUG(Dataset):
    def __init__(self, samples_json, img_size=512, out_size=128, cutout_size=64):
        with open(samples_json, 'r') as f:
            self.samples = json.load(f)

        self.img_size = img_size
        self.out_size = out_size
        self.cutout_size = cutout_size

        # Normalization params
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Color jitter for photometric augmentation
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        mask_path = sample['mask_path']
        class_id = sample['class_id']
        class_name = sample['class_name']

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Generate binary mask
        mask_np = np.array(mask)
        binary_mask = (mask_np == class_id).astype(np.uint8) * 255
        binary_mask = Image.fromarray(binary_mask)

        # Initial resize to working size
        image = F.resize(image, [self.img_size, self.img_size])
        binary_mask = F.resize(binary_mask, [self.img_size, self.img_size], interpolation=Image.NEAREST)

        # --- Augmentations ---
        # 1. Flips
        if random.random() < 0.5:
            image = F.hflip(image); binary_mask = F.hflip(binary_mask)
        if random.random() < 0.5:
            image = F.vflip(image); binary_mask = F.vflip(binary_mask)

        # 2. Random resized crop
        if random.random() < 0.5:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.7, 1.0), ratio=(0.75, 1.33)
            )
            image = F.resized_crop(image, i, j, h, w, [self.img_size, self.img_size])
            binary_mask = F.resized_crop(binary_mask, i, j, h, w, [self.img_size, self.img_size], interpolation=Image.NEAREST)

        # 3. Affine with shear
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            translate = (int(random.uniform(-0.05, 0.05) * self.img_size), int(random.uniform(-0.05, 0.05) * self.img_size))
            scale = random.uniform(0.8, 1.2)
            shear = random.uniform(-10, 10)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
            binary_mask = F.affine(binary_mask, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=Image.NEAREST)

        # 4. Photometric
        if random.random() < 0.7:
            image = self.color_jitter(image)
        if random.random() < 0.2:
            sigma = random.uniform(0.1, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        if random.random() < 0.2:
            b = random.uniform(0.8, 1.2)
            c = random.uniform(0.8, 1.2)
            image = F.adjust_brightness(image, b)
            image = F.adjust_contrast(image, c)

        # 5. CutOut
        if random.random() < 0.3:
            x0 = random.randint(0, self.img_size - self.cutout_size)
            y0 = random.randint(0, self.img_size - self.cutout_size)
            x1, y1 = x0 + self.cutout_size, y0 + self.cutout_size
            draw_img = ImageDraw.Draw(image)
            fill_color = tuple(int(m * 255) for m in self.mean)
            draw_img.rectangle([x0, y0, x1, y1], fill=fill_color)
            draw_mask = ImageDraw.Draw(binary_mask)
            draw_mask.rectangle([x0, y0, x1, y1], fill=0)

        # Ensure final sizes: image=img_size, mask=out_size
        image = F.resize(image, [self.img_size, self.img_size])
        binary_mask = F.resize(binary_mask, [self.out_size, self.out_size], interpolation=Image.NEAREST)

        # To tensor & normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        binary_mask = F.to_tensor(binary_mask)
        binary_mask = (binary_mask > 0).float()

        return {'image': image, 'label': binary_mask, 'text': class_name, 'fname': os.path.basename(img_path)}


class COCOBinaryMaskDataset_wNewAUG2(Dataset):
    def __init__(self, samples_json, img_size=512, out_size=128, cutout_size=64):
        with open(samples_json, 'r') as f:
            self.samples = json.load(f)

        self.img_size = img_size
        self.out_size = out_size
        self.cutout_size = cutout_size

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        mask_path = sample['mask_path']
        class_id = sample['class_id']
        class_name = sample['class_name']

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask_np = np.array(mask)
        binary_mask = (mask_np == class_id).astype(np.uint8) * 255
        binary_mask = Image.fromarray(binary_mask)

        # --- Augmentations ---

        # 0. RandomResizedCrop (handles resize jitter + safe crop)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.5, 2.0), ratio=(0.75, 1.33)
        )
        image = F.resized_crop(image, i, j, h, w, size=[self.img_size, self.img_size])
        binary_mask = F.resized_crop(binary_mask, i, j, h, w, size=[self.img_size, self.img_size], interpolation=Image.NEAREST)

        # 1. Flips
        if random.random() < 0.5:
            image = F.hflip(image); binary_mask = F.hflip(binary_mask)
        if random.random() < 0.5:
            image = F.vflip(image); binary_mask = F.vflip(binary_mask)

        # 2. Affine with shear
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            translate = (
                int(random.uniform(-0.05, 0.05) * self.img_size),
                int(random.uniform(-0.05, 0.05) * self.img_size)
            )
            scale = random.uniform(0.8, 1.2)
            shear = random.uniform(-10, 10)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
            binary_mask = F.affine(binary_mask, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=Image.NEAREST)

        # 3. Photometric
        if random.random() < 0.7:
            image = self.color_jitter(image)
        if random.random() < 0.2:
            sigma = random.uniform(0.1, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        if random.random() < 0.2:
            b = random.uniform(0.8, 1.2)
            c = random.uniform(0.8, 1.2)
            image = F.adjust_brightness(image, b)
            image = F.adjust_contrast(image, c)

        # 4. CutOut
        if random.random() < 0.3:
            cutout_frac = 0.2
            cutout_size = int(self.img_size * cutout_frac)
            x0 = random.randint(0, self.img_size - cutout_size)
            y0 = random.randint(0, self.img_size - cutout_size)
            x1, y1 = x0 + cutout_size, y0 + cutout_size
            draw_img = ImageDraw.Draw(image)
            fill_color = tuple(int(m * 255) for m in self.mean)
            draw_img.rectangle([x0, y0, x1, y1], fill=fill_color)
            draw_mask = ImageDraw.Draw(binary_mask)
            draw_mask.rectangle([x0, y0, x1, y1], fill=0)

        # Final resize to model-required output size
        binary_mask = F.resize(binary_mask, [self.out_size, self.out_size], interpolation=Image.NEAREST)

        # To tensor & normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        binary_mask = F.to_tensor(binary_mask)
        binary_mask = (binary_mask > 0).float()

        return {
            'image': image,
            'label': binary_mask,
            'text': class_name,
            'fname': os.path.basename(img_path)
        }
    
    
class UnlabeledSegmentationDataset(Dataset):
    def __init__(self, samples_json, img_size=512):
        with open(samples_json, 'r') as f:
            self.samples = json.load(f)

        self.img_size = img_size

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        class_name = sample.get('class_name', 'unknown')

        image = Image.open(img_path).convert("RGB")

        # === 1. Shared geometric augmentations ===
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.5, 2.0), ratio=(0.75, 1.33)
        )

        image = F.resized_crop(image, i, j, h, w, size=[self.img_size, self.img_size])
        # Random flip
        if random.random() < 0.5:
            image = F.hflip(image)
        if random.random() < 0.5:
            image = F.vflip(image)

        # Save two copies for weak and strong branches
        image_weak = image.copy()
        image_strong = image.copy()

        # === 2. Apply weak augmentations (very light) ===
        image_weak = F.to_tensor(image_weak)
        image_weak = F.normalize(image_weak, mean=self.mean, std=self.std)

        # === 3. Apply strong augmentations ===
        if random.random() < 0.7:
            image_strong = self.color_jitter(image_strong)
        if random.random() < 0.2:
            sigma = random.uniform(0.1, 1.5)
            image_strong = image_strong.filter(ImageFilter.GaussianBlur(radius=sigma))
        if random.random() < 0.2:
            b = random.uniform(0.8, 1.2)
            c = random.uniform(0.8, 1.2)
            image_strong = F.adjust_brightness(image_strong, b)
            image_strong = F.adjust_contrast(image_strong, c)
        if random.random() < 0.3:
            draw = ImageDraw.Draw(image_strong)
            cutout_size = int(self.img_size * 0.2)
            x0 = random.randint(0, self.img_size - cutout_size)
            y0 = random.randint(0, self.img_size - cutout_size)
            fill_color = tuple(int(m * 255) for m in self.mean)
            draw.rectangle([x0, y0, x0 + cutout_size, y0 + cutout_size], fill=fill_color)

        image_strong = F.to_tensor(image_strong)
        image_strong = F.normalize(image_strong, mean=self.mean, std=self.std)

        return {
            'image_weak': image_weak,
            'image_strong': image_strong,
            'text': class_name,
            'fname': os.path.basename(img_path)
        }

# ade20k_binary_mask_dataset.py

class ADE20KBinaryMaskDataset_wTextEmb(Dataset):
    """
    Each entry in samples_json = {
        "image_path": ".../ADE_train_00000001.jpg",
        "mask_path" : ".../ADE_train_00000001.png",
        "class_id"  : 12,
        "class_name": "cabinet"
    }
    One *object* == one sample.
    """
    def __init__(
        self,
        samples_json: str,
        img_size: int = 1024,
        out_size: int = 256,
        emb_dir : str | None = None           # dir containing "12_cabinet.pt", ...
    ):
        with open(samples_json, "r") as f:
            self.samples = json.load(f)

        self.img_size  = img_size
        self.out_size  = out_size

        # ---------- transforms ----------
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.transform_msk = transforms.Compose([
            transforms.Resize((out_size, out_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),                      # keeps {0,1}
        ])

        # ---------- text embeddings ----------
        if emb_dir is None:
            raise ValueError("Provide emb_dir so the dataset can attach CLIP text embeddings.")
        self.text_embs = {
            int(path.stem.split("_")[0]): torch.load(path, map_location="cpu")
            for path in Path(emb_dir).glob("*.pt")
        }
        if not self.text_embs:
            raise RuntimeError(f"No *.pt files found in {emb_dir}")
        
        self.class_id_list = sorted(self.text_embs.keys())


    # ---------------- Dataset API ----------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample      = self.samples[idx]
        img_path    = sample["image_path"]
        mask_path   = sample["mask_path"]
        class_id    = sample["class_id"]
        class_name  = sample["class_name"]

        # ----- image & binary mask -----
        image = Image.open(img_path).convert("RGB")

        full_mask = np.array(Image.open(mask_path))
        # ADE20K challenge masks: 0 = ignore, 1–150 = categories
        binary_mask = (full_mask == class_id).astype(np.uint8)

        image       = self.transform_img(image)                       # [3,H,W]
        binary_mask = self.transform_msk(Image.fromarray(binary_mask))
        binary_mask = (binary_mask > 0).float()                       # [1,H',W']


        # ----- text embedding -----
        try:
            text_emb = self.text_embs[class_id]                       # [512]
        except KeyError:
            raise KeyError(f"Missing embedding for class_id {class_id}")

        # Pick a random *other* class embedding
        if len(self.class_id_list) > 1:
            # exclude current class_id
            pool = [cid for cid in self.class_id_list if cid != class_id]
            other_id = np.random.choice(pool)
            other_text_emb = self.text_embs[other_id]
        else:
            # degenerate case: only one class in emb_dir
            other_text_emb = text_emb  # fallback


        return {
            "image": image,                 # [3, H, W]
            "label": binary_mask,           # [1, H', W']
            "text":  class_name,            # str (optional)
            "class_id": class_id,           # NEW: int
            "text_emb": text_emb,           # [D]
            "other_text_emb": other_text_emb,  # NEW: [D]
            "fname": os.path.basename(img_path),
        }


class ADE20KBinaryMaskDataset(Dataset):
    """
    Each entry in samples_json = {
        "image_path": ".../ADE_train_00000001.jpg",
        "mask_path" : ".../ADE_train_00000001.png",
        "class_id"  : 12,
        "class_name": "cabinet"
    }
    One *object* == one sample.
    """
    def __init__(
        self,
        samples_json: str,
        img_size: int = 1024,
        out_size: int = 256,
    ):
        with open(samples_json, "r") as f:
            self.samples = json.load(f)

        self.img_size  = img_size
        self.out_size  = out_size

        # ---------- transforms ----------
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.transform_msk = transforms.Compose([
            transforms.Resize((out_size, out_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),                      # keeps {0,1}
        ])


    # ---------------- Dataset API ----------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample      = self.samples[idx]
        img_path    = sample["image_path"]
        mask_path   = sample["mask_path"]
        class_id    = sample["class_id"]
        class_name  = sample["class_name"]

        # ----- image & binary mask -----
        image = Image.open(img_path).convert("RGB")

        full_mask = np.array(Image.open(mask_path))
        # ADE20K challenge masks: 0 = ignore, 1–150 = categories
        binary_mask = (full_mask == class_id).astype(np.uint8)

        image       = self.transform_img(image)                       # [3,H,W]
        binary_mask = self.transform_msk(Image.fromarray(binary_mask))
        binary_mask = (binary_mask > 0).float()                       # [1,H',W']

        return {
            "image"    : image,
            "label"    : binary_mask,
            "text"     : class_name,
            "fname"    : os.path.basename(img_path),
        }



import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision import transforms


class ADE20KBinaryMaskDataset_wNewAUG(Dataset):
    """
    Each entry in samples_json = {
        "image_path": ".../ADE_train_00000001.jpg",
        "mask_path" : ".../ADE_train_00000001.png",
        "class_id"  : 12,
        "class_name": "cabinet"
    }
    One *object* == one sample.
    """

    def __init__(self, samples_json, img_size=512, out_size=128, cutout_size=64):
        with open(samples_json, "r") as f:
            self.samples = json.load(f)

        self.img_size = img_size
        self.out_size = out_size
        self.cutout_size = cutout_size

        # Normalization params
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Color jitter for photometric augmentation
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image_path"]
        mask_path = sample["mask_path"]
        class_id = sample["class_id"]
        class_name = sample["class_name"]

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        full_mask = np.array(Image.open(mask_path))
        binary_mask = (full_mask == class_id).astype(np.uint8) * 255
        binary_mask = Image.fromarray(binary_mask)

        # Initial resize
        image = F.resize(image, [self.img_size, self.img_size])
        binary_mask = F.resize(
            binary_mask, [self.img_size, self.img_size], interpolation=Image.NEAREST
        )

        # --- Augmentations ---
        # 1. Flips
        if random.random() < 0.5:
            image = F.hflip(image)
            binary_mask = F.hflip(binary_mask)
        if random.random() < 0.5:
            image = F.vflip(image)
            binary_mask = F.vflip(binary_mask)

        # 2. Random resized crop
        if random.random() < 0.5:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.7, 1.0), ratio=(0.75, 1.33)
            )
            image = F.resized_crop(image, i, j, h, w, [self.img_size, self.img_size])
            binary_mask = F.resized_crop(
                binary_mask, i, j, h, w, [self.img_size, self.img_size],
                interpolation=Image.NEAREST
            )

        # 3. Affine with shear
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            translate = (
                int(random.uniform(-0.05, 0.05) * self.img_size),
                int(random.uniform(-0.05, 0.05) * self.img_size),
            )
            scale = random.uniform(0.8, 1.2)
            shear = random.uniform(-10, 10)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
            binary_mask = F.affine(
                binary_mask, angle=angle, translate=translate, scale=scale, shear=shear,
                interpolation=Image.NEAREST
            )

        # 4. Photometric
        if random.random() < 0.7:
            image = self.color_jitter(image)
        if random.random() < 0.2:
            sigma = random.uniform(0.1, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        if random.random() < 0.2:
            b = random.uniform(0.8, 1.2)
            c = random.uniform(0.8, 1.2)
            image = F.adjust_brightness(image, b)
            image = F.adjust_contrast(image, c)

        # 5. CutOut
        if random.random() < 0.3:
            x0 = random.randint(0, self.img_size - self.cutout_size)
            y0 = random.randint(0, self.img_size - self.cutout_size)
            x1, y1 = x0 + self.cutout_size, y0 + self.cutout_size
            draw_img = ImageDraw.Draw(image)
            fill_color = tuple(int(m * 255) for m in self.mean)
            draw_img.rectangle([x0, y0, x1, y1], fill=fill_color)
            draw_mask = ImageDraw.Draw(binary_mask)
            draw_mask.rectangle([x0, y0, x1, y1], fill=0)

        # Final resizing
        image = F.resize(image, [self.img_size, self.img_size])
        binary_mask = F.resize(binary_mask, [self.out_size, self.out_size], interpolation=Image.NEAREST)

        # To tensor & normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        binary_mask = F.to_tensor(binary_mask)
        binary_mask = (binary_mask > 0).float()

        return {
            "image": image,
            "label": binary_mask,
            "text": class_name,
            "fname": os.path.basename(img_path),
        }


class PascalVOCBinaryMaskDataset_wNewAUG(Dataset):
    """
    Each entry in samples_json = {
        "image_path": ".../JPEGImages/2010_004493.jpg",
        "mask_path" : ".../SegmentationClass/2010_004493.png",
        "class_id"  : 7,
        "class_name": "car"
    }
    One *object* == one sample.
    """

    def __init__(self, samples_json, img_size=512, out_size=128, cutout_size=64):
        with open(samples_json, "r") as f:
            self.samples = json.load(f)

        self.img_size = img_size
        self.out_size = out_size
        self.cutout_size = cutout_size

        # Normalization params
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Color jitter for photometric augmentation
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image_path"]
        mask_path = sample["mask_path"]
        class_id = sample["class_id"]
        class_name = sample["class_name"]

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        full_mask = np.array(Image.open(mask_path))
        binary_mask = (full_mask == class_id).astype(np.uint8) * 255
        binary_mask = Image.fromarray(binary_mask)

        # Initial resize
        image = F.resize(image, [self.img_size, self.img_size])
        binary_mask = F.resize(
            binary_mask, [self.img_size, self.img_size], interpolation=Image.NEAREST
        )

        # --- Augmentations ---
        # 1. Flips
        if random.random() < 0.5:
            image = F.hflip(image)
            binary_mask = F.hflip(binary_mask)
        if random.random() < 0.5:
            image = F.vflip(image)
            binary_mask = F.vflip(binary_mask)

        # 2. Random resized crop
        if random.random() < 0.5:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.7, 1.0), ratio=(0.75, 1.33)
            )
            image = F.resized_crop(image, i, j, h, w, [self.img_size, self.img_size])
            binary_mask = F.resized_crop(
                binary_mask, i, j, h, w, [self.img_size, self.img_size],
                interpolation=Image.NEAREST
            )

        # 3. Affine with shear
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            translate = (
                int(random.uniform(-0.05, 0.05) * self.img_size),
                int(random.uniform(-0.05, 0.05) * self.img_size),
            )
            scale = random.uniform(0.8, 1.2)
            shear = random.uniform(-10, 10)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
            binary_mask = F.affine(
                binary_mask, angle=angle, translate=translate, scale=scale, shear=shear,
                interpolation=Image.NEAREST
            )

        # 4. Photometric
        if random.random() < 0.7:
            image = self.color_jitter(image)
        if random.random() < 0.2:
            sigma = random.uniform(0.1, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        if random.random() < 0.2:
            b = random.uniform(0.8, 1.2)
            c = random.uniform(0.8, 1.2)
            image = F.adjust_brightness(image, b)
            image = F.adjust_contrast(image, c)

        # 5. CutOut
        if random.random() < 0.3:
            x0 = random.randint(0, self.img_size - self.cutout_size)
            y0 = random.randint(0, self.img_size - self.cutout_size)
            x1, y1 = x0 + self.cutout_size, y0 + self.cutout_size
            draw_img = ImageDraw.Draw(image)
            fill_color = tuple(int(m * 255) for m in self.mean)
            draw_img.rectangle([x0, y0, x1, y1], fill=fill_color)
            draw_mask = ImageDraw.Draw(binary_mask)
            draw_mask.rectangle([x0, y0, x1, y1], fill=0)

        # Final resizing
        image = F.resize(image, [self.img_size, self.img_size])
        binary_mask = F.resize(binary_mask, [self.out_size, self.out_size], interpolation=Image.NEAREST)

        # To tensor & normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        binary_mask = F.to_tensor(binary_mask)
        binary_mask = (binary_mask > 0).float()

        return {
            "image": image,
            "label": binary_mask,
            "text": class_name,
            "fname": os.path.basename(img_path),
        }

import os, json
from typing import Optional, Dict, List, Sequence, Callable, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

SynSource = Union[Dict[str, Sequence[str]], Callable[[str], Sequence[str]]]

class PascalVOCBinaryMaskDataset(Dataset):
    """
    Each entry in samples_json = {
        "image_path": ".../JPEGImages/2010_004493.jpg",
        "mask_path" : ".../SegmentationClass/2010_004493.png",
        "class_id"  : 7,
        "class_name": "car"
    }
    One *object* == one sample.
    """
    def __init__(
        self,
        samples_json: str,
        img_size: int = 1024,
        out_size: int = 256,
        use_syn: bool = False,
        synonyms = None,
        include_self: bool = False,     # include the class_name itself in the synonyms list
        lowercase: bool = True,        # normalize synonyms to lowercase
        dedup: bool = False             # remove duplicates while preserving order
    ):
        with open(samples_json, "r") as f:
            self.samples = json.load(f)

        self.img_size   = img_size
        self.out_size   = out_size
        self.use_syn    = use_syn
        self.include_self = include_self
        self.lowercase  = lowercase
        self.dedup      = dedup

        # If no synonyms passed, you can define a lightweight default dict here.
        # Keep it minimal; pass your own full mapping for production.
        self._default_synonyms: Dict[str, List[str]] = {
            "aeroplane": ["aeroplane","airplane", "plane", "aircraft", "glider"],
            "bicycle": ["bicycle","tricycle", "unicycle"],
            "bird": ["bird"],
            "boat": ["boat","ship", "vessel", "rowing boat", "pedalo"],
            "bottle": ["bottle","plastic bottle", "glass bottle", "feeding bottle"], #"flask", "container"
            "bus": ["bus","minibus"],
            "car": ["car","van", "large family car", "realistic toy car"], # sedan
            "cat": ["cat","domestic cat"],
            "chair": ["chair","armchair","deckchair"],
            "cow": ["cow"],
            "diningtable": ["dining table","table for eating at"],
            "dog": ["dog","domestic dog"],
            "horse": ["horse","pony","donkey","mule"],
            "motorbike": ["motorbike","moped", "scooter", "sidecar"],
            "person": ["person","people", "baby", "face"], # "human"
            "pottedplant": ["potted plant","indoor plant in a pot", "outdoor plant in a pot"],
            "sheep": ["sheep"],
            "sofa": ["sofa"],
            "train": ["train","train carriage"],
            "tvmonitor": ["tv","standalone screen", "monitor"],
        }

        # `synonyms` can be a dict or a callable; if None, use the default dict.
        self.synonyms = synonyms if synonyms is not None else self._default_synonyms

        # ---------- transforms ----------
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.transform_msk = transforms.Compose([
            transforms.Resize((out_size, out_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Precompute texts once for speed if using synonyms (O(N))
        if self.use_syn:
            self._texts: List[Union[str, List[str]]] = [
                self._build_text_for(s["class_name"]) for s in self.samples
            ]
        else:
            # keep the original string form for compatibility
            self._texts = [s["class_name"] for s in self.samples]

    # ---------------- helpers ----------------
    def _normalize_tokens(self, tokens: Sequence[str]) -> List[str]:
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        if self.dedup:
            # deduplicate while preserving order
            tokens = list(dict.fromkeys(tokens))
        return list(tokens)

    def _lookup_synonyms(self, class_name: str) -> List[str]:
        """Return synonyms for a class name from dict or callable; [] if none."""
        if callable(self.synonyms):
            try:
                syns = list(self.synonyms(class_name))
            except Exception:
                syns = []
        else:
            # Dict lookup: be lenient with case
            key = class_name
            syns = self.synonyms.get(key, [])
            if not syns:
                syns = self.synonyms.get(key.lower(), [])
        return list(syns)

    def _build_text_for(self, class_name: str) -> List[str]:
        syns = self._lookup_synonyms(class_name)
        out = []
        if self.include_self:
            out.append(class_name)
        out.extend(syns)
        out = self._normalize_tokens(out)
        # Guarantee at least the class name
        if len(out) == 0:
            out = [class_name.lower() if self.lowercase else class_name]
        return out

    # ---------------- Dataset API ----------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample      = self.samples[idx]
        img_path    = sample["image_path"]
        mask_path   = sample["mask_path"]
        class_id    = sample["class_id"]

        # ----- image & binary mask -----
        image = Image.open(img_path).convert("RGB")

        full_mask = np.array(Image.open(mask_path))
        binary_mask = (full_mask == class_id).astype(np.uint8)

        image       = self.transform_img(image)                       # [3,H,W]
        binary_mask = self.transform_msk(Image.fromarray(binary_mask))
        binary_mask = (binary_mask > 0).float()                       # [1,H',W']

        # Precomputed in __init__ for speed; either str or List[str]
        text_out = self._texts[idx]

        return {
            "image" : image,
            "label" : binary_mask,
            "text"  : text_out,  # str if use_syn=False, List[str] if use_syn=True
            "fname" : os.path.basename(img_path),
        }


from typing import Optional, Dict, List, Sequence, Callable, Union
import os, json, random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

# ---- Text handling types ----
SynSource = Union[Dict[str, Sequence[str]], Callable[[str], Sequence[str]]]

class PascalVOCBinaryMaskDatasetUnified(Dataset):
    """
    Unified Pascal VOC single-class (binary) mask dataset with:
      - Optional heavy augmentations (use_aug=True)
      - Text/synonym handling identical to your base dataset

    Each entry in samples_json:
      {
        "image_path": ".../JPEGImages/2010_004493.jpg",
        "mask_path" : ".../SegmentationClass/2010_004493.png",
        "class_id"  : 7,
        "class_name": "car"
      }

    TEXT OUTPUT LOGIC
    -----------------
    - If use_syn=False:   "text" is a str (the original class_name).
    - If use_syn=True:    "text" is a List[str] (synonyms, optionally incl. class_name).
                          Control with include_self/lowercase/dedup.
    """
    def __init__(
        self,
        samples_json: str,
        img_size: int = 512,
        out_size: int = 128,
        # ---- augmentation switches/params ----
        use_aug: bool = True,
        cutout_size: int = 64,
        # ---- text/synonym handling ----
        use_syn: bool = False,
        synonyms=None,
        include_self: bool = False,   # include the class_name itself in the synonyms list
        lowercase: bool = True,       # normalize synonyms to lowercase
        dedup: bool = False           # remove duplicates while preserving order
    ):
        with open(samples_json, "r") as f:
            self.samples = json.load(f)

        # shapes
        self.img_size    = img_size
        self.out_size    = out_size

        # aug
        self.use_aug     = use_aug
        self.cutout_size = cutout_size

        # normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]

        # photometric aug (used only when use_aug=True)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

        # ---- text handling config ----
        self.use_syn      = use_syn
        self.include_self = include_self
        self.lowercase    = lowercase
        self.dedup        = dedup

        # small default synonyms map; pass your own for production
        self._default_synonyms: Dict[str, List[str]] = {
            "aeroplane":   ["aeroplane", "airplane", "plane", "aircraft", "glider"],
            "bicycle":     ["bicycle", "tricycle", "unicycle"],
            "bird":        ["bird"],
            "boat":        ["boat", "ship", "vessel", "rowing boat", "pedalo"],
            "bottle":      ["bottle", "plastic bottle", "glass bottle", "feeding bottle"],
            "bus":         ["bus", "minibus"],
            "car":         ["car", "van", "large family car", "realistic toy car"],
            "cat":         ["cat", "domestic cat"],
            "chair":       ["chair", "armchair", "deckchair"],
            "cow":         ["cow"],
            "diningtable": ["dining table", "table for eating at"],
            "dog":         ["dog", "domestic dog"],
            "horse":       ["horse", "pony", "donkey", "mule"],
            "motorbike":   ["motorbike", "moped", "scooter", "sidecar"],
            "person":      ["person", "people", "baby", "face"],
            "pottedplant": ["potted plant", "indoor plant in a pot", "outdoor plant in a pot"],
            "sheep":       ["sheep"],
            "sofa":        ["sofa"],
            "train":       ["train", "train carriage"],
            "tvmonitor":   ["tv", "standalone screen", "monitor"],
        }
        # self.synonyms: SynSource = synonyms if synonyms is not None else self._default_synonyms
        self.synonyms = synonyms if synonyms is not None else self._default_synonyms

        # simple (no-aug) transforms
        self._simple_img = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        self._simple_msk = transforms.Compose([
            transforms.Resize((self.out_size, self.out_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Precompute texts once (O(N)) for speed if using synonyms
        if self.use_syn:
            self._texts: List[Union[str, List[str]]] = [
                self._build_text_for(s["class_name"]) for s in self.samples
            ]
        else:
            self._texts = [s["class_name"] for s in self.samples]

    # ---------------- text helpers ----------------
    def _normalize_tokens(self, tokens: Sequence[str]) -> List[str]:
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        if self.dedup:
            tokens = list(dict.fromkeys(tokens))  # preserve order
        return list(tokens)

    def _lookup_synonyms(self, class_name: str) -> List[str]:
        if callable(self.synonyms):
            try:
                syns = list(self.synonyms(class_name))
            except Exception:
                syns = []
        else:
            key = class_name
            syns = self.synonyms.get(key, [])
            if not syns:
                syns = self.synonyms.get(key.lower(), [])
        return list(syns)

    def _build_text_for(self, class_name: str) -> List[str]:
        syns = self._lookup_synonyms(class_name)
        out: List[str] = []
        if self.include_self:
            out.append(class_name)
        out.extend(syns)
        out = self._normalize_tokens(out)
        if len(out) == 0:
            out = [class_name.lower() if self.lowercase else class_name]
        return out

    # ---------------- Dataset API ----------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img_path   = s["image_path"]
        mask_path  = s["mask_path"]
        class_id   = s["class_id"]
        class_name = s["class_name"]

        # load
        image = Image.open(img_path).convert("RGB")
        full_mask = np.array(Image.open(mask_path))
        binary_mask = (full_mask == class_id).astype(np.uint8) * 255
        binary_mask = Image.fromarray(binary_mask)

        if not self.use_aug:
            # --- simple, deterministic path ---
            image_t = self._simple_img(image)                      # [3,H,W]
            mask_t  = self._simple_msk(binary_mask)                # [1,H',W']
            mask_t  = (mask_t > 0).float()
        else:
            # --- augmentation path (mirrors your wNewAUG) ---
            # initial resize
            image = F.resize(image, [self.img_size, self.img_size])
            binary_mask = F.resize(binary_mask, [self.img_size, self.img_size], interpolation=Image.NEAREST)

            # 1) flips
            if random.random() < 0.5:
                image = F.hflip(image)
                binary_mask = F.hflip(binary_mask)
            if random.random() < 0.5:
                image = F.vflip(image)
                binary_mask = F.vflip(binary_mask)

            # 2) random resized crop
            if random.random() < 0.5:
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    image, scale=(0.7, 1.0), ratio=(0.75, 1.33)
                )
                image = F.resized_crop(image, i, j, h, w, [self.img_size, self.img_size])
                binary_mask = F.resized_crop(
                    binary_mask, i, j, h, w, [self.img_size, self.img_size],
                    interpolation=Image.NEAREST
                )

            # 3) affine with shear
            if random.random() < 0.5:
                angle = random.uniform(-30, 30)
                translate = (
                    int(random.uniform(-0.05, 0.05) * self.img_size),
                    int(random.uniform(-0.05, 0.05) * self.img_size),
                )
                scale = random.uniform(0.8, 1.2)
                shear = random.uniform(-10, 10)
                image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
                binary_mask = F.affine(
                    binary_mask, angle=angle, translate=translate, scale=scale, shear=shear,
                    interpolation=Image.NEAREST
                )

            # 4) photometric
            if random.random() < 0.7:
                image = self.color_jitter(image)
            if random.random() < 0.2:
                sigma = random.uniform(0.1, 1.5)
                image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
            if random.random() < 0.2:
                b = random.uniform(0.8, 1.2)
                c = random.uniform(0.8, 1.2)
                image = F.adjust_brightness(image, b)
                image = F.adjust_contrast(image, c)

            # 5) CutOut
            if random.random() < 0.3:
                x0 = random.randint(0, self.img_size - self.cutout_size)
                y0 = random.randint(0, self.img_size - self.cutout_size)
                x1, y1 = x0 + self.cutout_size, y0 + self.cutout_size
                draw_img = ImageDraw.Draw(image)
                fill_color = tuple(int(m * 255) for m in self.mean)
                draw_img.rectangle([x0, y0, x1, y1], fill=fill_color)
                draw_mask = ImageDraw.Draw(binary_mask)
                draw_mask.rectangle([x0, y0, x1, y1], fill=0)

            # final resizing
            image = F.resize(image, [self.img_size, self.img_size])
            binary_mask = F.resize(binary_mask, [self.out_size, self.out_size], interpolation=Image.NEAREST)

            # to tensor & normalize
            image_t = F.to_tensor(image)
            image_t = F.normalize(image_t, mean=self.mean, std=self.std)
            mask_t  = F.to_tensor(binary_mask)
            mask_t  = (mask_t > 0).float()

        # choose text output (precomputed if use_syn)
        text_out = self._texts[idx] if self.use_syn else class_name

        return {
            "image": image_t,             # [3, H, W]
            "label": mask_t,              # [1, H', W']
            "text":  text_out,            # str or List[str]
            "fname": os.path.basename(img_path),
        }


import json, os, re
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class COD10KBinaryMaskDataset_wTextEmb(Dataset):
    r"""
    One *image* == one sample.
    We assume the standard COD10K folder layout:

        <data_path>/
            COD10K/
                Train/
                    Image/       *.jpg
                    GT_Object/   *.png | *.jpg
                Test/
                    Image/
                    GT_Object/
                Val/             # (if you keep a validation split)

    Class name is parsed from the filename, e.g.
        COD10K-CAM-1-Aquatic-1-BatFish-1.jpg  ->  "BatFish"
    """

    def __init__(
        self,
        data_path: str = '../Datasets/',
        split: Literal["Train", "Test", "Val"] = "Train",
        img_size: int = 1024,
        out_size: int = 256,
        emb_dir: str | None = None,           # "/.../clip_text_embs/BatFish.pt", ...
    ):
        super().__init__()

        # ---------- collect paired image / mask paths ----------
        img_root = Path(data_path) / "COD10K" / split / "Image"
        msk_root = Path(data_path) / "COD10K" / split / "GT_Object"

        if not (img_root.is_dir() and msk_root.is_dir()):
            raise FileNotFoundError(f"Expecting COD10K under {img_root.parent}, "
                                    "but the folder(s) were not found.")

        self.samples: list[dict] = []
        for img_path in sorted(img_root.glob("*.jpg")):
            fname = img_path.name
            base  = fname.rsplit(".", 1)[0]                     # without suffix
            msk_path = msk_root / f"{base}.png"                 # masks are *.png
            if not msk_path.exists():                           # fallback to *.jpg
                msk_path = msk_root / f"{base}.jpg"
            if not msk_path.exists():
                continue  # skip if no mask

            # ---- parse class name from the filename ----
            # pattern "...-<class>-<number>.<ext>"
            m = re.search(r"-(?P<class>[^-]+)-\d+(?:\.\w+)$", fname)
            if not m:
                raise ValueError(f"Could not parse class name from {fname}")
            class_name = m.group("class")

            self.samples.append({
                "image_path" : str(img_path),
                "mask_path"  : str(msk_path),
                "class_name" : class_name,
            })

        if not self.samples:
            raise RuntimeError("No *.jpg images paired with masks were found "
                               f"in {img_root}")

        # ---------- transforms ----------
        self.tr_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.tr_msk = transforms.Compose([
            transforms.Resize((out_size, out_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),                      # keeps {0,1}
        ])

        # ---------- (optional) CLIP text embeddings ----------
        if emb_dir is not None:
            self.text_embs = {
                Path(p).stem: torch.load(p, map_location="cpu")
                for p in Path(emb_dir).glob("*.pt")
            }
            if not self.text_embs:
                raise RuntimeError(f"No *.pt files found in {emb_dir}")
        else:
            self.text_embs = None  # user may choose to skip embeddings

    # ============== Dataset API ==============
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s          = self.samples[idx]
        img_path   = s["image_path"]
        msk_path   = s["mask_path"]
        class_name = s["class_name"]

        # -- image & binary mask --
        image = Image.open(img_path).convert("RGB")

        # GT masks in COD10K are binary already (0 / 255)
        binary_mask = np.array(Image.open(msk_path).convert("L"))
        binary_mask = (binary_mask > 127).astype(np.uint8)          # 0/1

        image       = self.tr_img(image)                            # [3,H,W]
        binary_mask = self.tr_msk(Image.fromarray(binary_mask*255))
        binary_mask = (binary_mask > 0).float()                     # [1,H',W']

        # -- text embedding (optional) --
        if self.text_embs is not None:
            try:
                text_emb = self.text_embs[class_name]
            except KeyError:
                raise KeyError(f"Missing embedding file for class '{class_name}' "
                               "in emb_dir")
        else:
            text_emb = None

        return {
            "image"    : image,
            "label"    : binary_mask,
            "text"     : class_name,
            "text_emb" : text_emb,          # may be None if emb_dir was None
            "fname"    : os.path.basename(img_path),
        }
    

import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CamouflagedBinaryMaskDataset_wTextEmb(Dataset):
    r"""
    Class name from filename:
      • COD10K-CAM-...-<Superclass>-...-<Class>-<id>.<ext> -> <Class>
      • COD10K ... with Superclass=="Other" or Class=="Other" -> "Camouflaged object"
      • camourflage_#####.<ext> -> "Camouflaged object"
      • animal-#####.<ext> -> "Camouflaged animal"
      • anything else -> "Camouflaged object"
    """

    _RE_COD10K = re.compile(
        r'^COD10K-CAM-\d+-(?P<superclass>[^-]+)-\d+-(?P<classname>[^-]+)-\d+\.(?:jpg|jpeg|png|bmp|tif|tiff)$',
        re.IGNORECASE
    )
    _RE_ANIMAL = re.compile(  # animal-22.jpg
        r'^(?:animal|animals)-\d+\.(?:jpg|jpeg|png|bmp|tif|tiff)$',
        re.IGNORECASE
    )
    _RE_CAMO = re.compile(    # camou?rflage_00002.jpg
        r'^camou?rflage_\d+\.(?:jpg|jpeg|png|bmp|tif|tiff)$',
        re.IGNORECASE
    )

    def __init__(
        self,
        data_path: str = '../Datasets/',
        split: Literal["TrainDataset", "TestDataset", "Val"] = "TrainDataset",
        img_size: int = 1024,
        out_size: int = 256,
        emb_dir: str | None = None,
    ):
        super().__init__()

        img_root = Path(data_path) / "Combined_Camouflaged" / split / "Imgs"
        msk_root = Path(data_path) / "Combined_Camouflaged" / split / "GT"
        if not (img_root.is_dir() and msk_root.is_dir()):
            raise FileNotFoundError(
                f"Expecting dataset under {img_root.parent}, but {img_root} or {msk_root} was not found."
            )

        exts = ("*.jpg", "*.jpeg", "*.png")
        img_paths = []
        for pat in exts:
            img_paths.extend(sorted(img_root.glob(pat)))

        self.samples: list[dict] = []
        for img_path in img_paths:
            fname = img_path.name
            base = fname.rsplit(".", 1)[0]
            msk_path = msk_root / f"{base}.png"
            if not msk_path.exists():
                alt = msk_root / f"{base}.jpg"
                if alt.exists():
                    msk_path = alt
                else:
                    continue  # skip if no mask

            class_name = self._parse_class_from_filename(fname)

            self.samples.append({
                "image_path": str(img_path),
                "mask_path": str(msk_path),
                "class_name": class_name,
            })

        if not self.samples:
            raise RuntimeError(f"No images paired with masks were found in {img_root}")

        # ---------- transforms ----------
        self.tr_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.tr_msk = transforms.Compose([
            transforms.Resize((out_size, out_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # keeps {0,1}
        ])

        self.emb_dir = Path(emb_dir) if emb_dir is not None else None
        if self.emb_dir is not None:
            self.text_embs = {}
            for p in self.emb_dir.glob("*.pt"):
                key = self._norm_label(p.stem)          # <-- normalized key
                self.text_embs[key] = torch.load(p, map_location="cpu")
            if not self.text_embs:
                raise RuntimeError(f"No *.pt files found in {self.emb_dir}")
        else:
            self.text_embs = None


    # ---------- helpers ----------
    @staticmethod
    def _norm_label(name: str) -> str:
        # unify spaces/underscores and case
        import re
        return re.sub(r'[\s_]+', '_', name.strip()).lower()

    @classmethod
    def _parse_class_from_filename(cls, fname: str) -> str:
        # COD10K format
        m = cls._RE_COD10K.match(fname)
        if m:
            superclass = m.group("superclass").strip().lower()
            classname  = m.group("classname").strip().lower()
            if superclass == "other" or classname == "other":
                return "Camouflaged object"
            return m.group("classname").strip()  # preserve original case

        # animal-#####
        if cls._RE_ANIMAL.match(fname):
            return "Camouflaged animal"

        # camou(r)flage_#####
        if cls._RE_CAMO.match(fname):
            return "Camouflaged object"

        # fallback
        return "Camouflaged object"

    # ============== Dataset API ==============
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img_path = s["image_path"]
        msk_path = s["mask_path"]
        class_name = s["class_name"]

        image = Image.open(img_path).convert("RGB")

        binary_mask = np.array(Image.open(msk_path).convert("L"))
        binary_mask = (binary_mask > 127).astype(np.uint8)

        image = self.tr_img(image)
        binary_mask = self.tr_msk(Image.fromarray(binary_mask * 255))
        binary_mask = (binary_mask > 0).float()

        # text embedding (optional)
        if self.text_embs is not None:
            key = self._norm_label(class_name)
            try:
                text_emb = self.text_embs[key]
            except KeyError as e:
                expected = class_name.replace(" ", "_") + ".pt"
                raise KeyError(
                    f"Missing embedding for class '{class_name}'. "
                    f"Expected a file like '{expected}' in {self.emb_dir}."
                ) from e
        else:
            text_emb = None

        return {
            "image": image,
            "label": binary_mask,
            "text": class_name,    # e.g., "LeafySeaDragon", "Camouflaged object", "Camouflaged animal"
            "text_emb": text_emb,
            "fname": os.path.basename(img_path),
        }


class CropsBinaryMaskDataset_wNewAUG(Dataset):
    def __init__(self, samples_json, img_size=512, out_size=128, cutout_size=64, use_aug=True):
        with open(samples_json, 'r') as f:
            self.samples = json.load(f)

        self.img_size = img_size
        self.out_size = out_size
        self.cutout_size = cutout_size
        self.use_aug = use_aug

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        mask_path = sample['mask_path']
        class_id = sample['class_id']
        class_name = sample['class_name']

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask_np = np.array(mask)
        binary_mask = (mask_np > 0).astype(np.uint8) * 255
        binary_mask = Image.fromarray(binary_mask)

        if self.use_aug:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.5, 2.0), ratio=(0.75, 1.33)
            )
            image = F.resized_crop(image, i, j, h, w, size=[self.img_size, self.img_size])
            binary_mask = F.resized_crop(binary_mask, i, j, h, w, size=[self.img_size, self.img_size], interpolation=Image.NEAREST)

            if random.random() < 0.5:
                image = F.hflip(image)
                binary_mask = F.hflip(binary_mask)
            if random.random() < 0.5:
                image = F.vflip(image)
                binary_mask = F.vflip(binary_mask)

            if random.random() < 0.5:
                angle = random.uniform(-30, 30)
                translate = (
                    int(random.uniform(-0.05, 0.05) * self.img_size),
                    int(random.uniform(-0.05, 0.05) * self.img_size)
                )
                scale = random.uniform(0.8, 1.2)
                shear = random.uniform(-10, 10)
                image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=shear)
                binary_mask = F.affine(binary_mask, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=Image.NEAREST)

            if random.random() < 0.7:
                image = self.color_jitter(image)
            if random.random() < 0.2:
                sigma = random.uniform(0.1, 1.5)
                image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
            if random.random() < 0.2:
                b = random.uniform(0.8, 1.2)
                c = random.uniform(0.8, 1.2)
                image = F.adjust_brightness(image, b)
                image = F.adjust_contrast(image, c)

            if random.random() < 0.3:
                cutout_frac = 0.2
                cutout_size = int(self.img_size * cutout_frac)
                x0 = random.randint(0, self.img_size - cutout_size)
                y0 = random.randint(0, self.img_size - cutout_size)
                x1, y1 = x0 + cutout_size, y0 + cutout_size
                draw_img = ImageDraw.Draw(image)
                fill_color = tuple(int(m * 255) for m in self.mean)
                draw_img.rectangle([x0, y0, x1, y1], fill=fill_color)
                draw_mask = ImageDraw.Draw(binary_mask)
                draw_mask.rectangle([x0, y0, x1, y1], fill=0)
        else:
            image = F.resize(image, [self.img_size, self.img_size])
            binary_mask = F.resize(binary_mask, [self.img_size, self.img_size], interpolation=Image.NEAREST)

        binary_mask = F.resize(binary_mask, [self.out_size, self.out_size], interpolation=Image.NEAREST)

        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)

        binary_mask = F.to_tensor(binary_mask)
        binary_mask = (binary_mask > 0).float()

        return {
            'image': image,
            'label': binary_mask,
            'text': class_name,
            'fname': os.path.basename(img_path),
            'class_id': class_id
        }