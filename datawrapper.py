from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils, datasets
from PIL import Image


class dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.info.iloc[idx, 0])

        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image


class concattraindataset(Dataset):
    def __init__(self, *sets):
        self.sets = sets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.sets)

    def __len__(self):
        return min(len(d) for d in self.sets)
        # return 80000


class concattestdataset(Dataset):
    def __init__(self, *sets):
        self.sets = sets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.sets)

    def __len__(self):
        return 8000


def trainloader(config):

    content_transform = transforms.Compose([transforms.Resize(config.contentSize),
                                            transforms.RandomCrop(config.finalSize),
                                            transforms.ToTensor()])
    style_transform = transforms.Compose([transforms.Resize(config.styleSize),
                                          transforms.RandomCrop(config.finalSize),
                                          transforms.ToTensor()])

    contentDir = os.path.join(config.contentDir, 'train2014_sliced')
    contentFile = os.path.join(config.contentDir, 'train2014_sliced.csv')
    styleDir = os.path.join(config.styleDir, 'train_sliced')
    styleFile = os.path.join(config.styleDir, 'train_sliced.csv')

    train_loader = DataLoader(
        concattraindataset(dataset(contentFile, contentDir, transform=content_transform),
                      dataset(styleFile, styleDir, transform=style_transform)),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2)

    return train_loader


def testloader(config):

    content_transform = transforms.Compose([transforms.Resize(config.contentSize),
                                            transforms.RandomCrop(config.finalSize),
                                            transforms.ToTensor()])
    style_transform = transforms.Compose([transforms.Resize(config.styleSize),
                                          transforms.RandomCrop(config.finalSize),
                                          transforms.ToTensor()])

    contentDir = os.path.join(config.contentDir, 'test2014_sliced')
    contentFile = os.path.join(config.contentDir, 'test2014_sliced.csv')
    styleDir = os.path.join(config.styleDir, 'test_sliced')
    styleFile = os.path.join(config.styleDir, 'test_sliced.csv')

    train_loader = DataLoader(
        concattestdataset(dataset(contentFile, contentDir, transform=content_transform),
                      dataset(styleFile, styleDir, transform=style_transform)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2)

    return train_loader


# class Rescale(object):
#     """Rescale the image in a sample to a given size.
#
#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, int)
#         self.output_size = output_size
#
#     def __call__(self, image):
#
#         image = np.asarray(image)
#
#         h, w = image.shape[:2]
#         if h > w:
#             new_h, new_w = self.output_size * h / w, self.output_size
#         else:
#             new_h, new_w = self.output_size, self.output_size * w / h
#
#         new_h, new_w = int(new_h), int(new_w)
#
#         image = transform.resize(image, (new_h, new_w))
#
#         return image


# class RandomCrop(object):
#     """Crop randomly the image in a sample.
#
#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, int)
#         self.output_size = (output_size, output_size)
#
#     def __call__(self, image):
#         h, w = image.shape[:2]
#         new_h, new_w = self.output_size
#
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)
#
#         image = image[top: top + new_h, left: left + new_w]
#
#         return image


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, image):
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C x H x W
#
#         # try:
#         #     if len(image.shape) < 3:
#         #         image = np.stack((image,) * 3, axis=-1)
#         #         assert(len(image.shape) == 3)
#         #         assert (image.shape[2] == 3)
#         #         assert (image.shape == torch.Size([3, 256, 256]))
#         # except:
#         #     return torch.rand((3, 256, 256), dtype=torch.float32)
#
#         return torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32))


# def trainloader(config):
#
#     content_transform = transforms.Compose([Rescale(config.contentSize),
#                                             RandomCrop(config.finalSize),
#                                             ToTensor()])
#
#     style_transform = transforms.Compose([Rescale(config.styleSize),
#                                           RandomCrop(config.finalSize),
#                                           ToTensor()])
#
#     contentDir = os.path.join(config.contentDir, 'train2014_24images')
#     contentFile = os.path.join(config.contentDir, 'train2014_24images.csv')
#     styleDir = os.path.join(config.styleDir, 'train_24images')
#     styleFile = os.path.join(config.styleDir, 'train_24images.csv')
#
#     content_loader = DataLoader(
#         contentDataset(contentFile, contentDir, transform=content_transform),
#         batch_size=config.batch_size,
#         shuffle=True,
#         num_workers=2)
#     style_loader = DataLoader(
#         contentDataset(styleFile, styleDir, transform=style_transform),
#         batch_size=config.batch_size,
#         shuffle=True,
#         num_workers=2)
#
#     return content_loader, style_loader
