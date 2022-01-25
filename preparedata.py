from __future__ import print_function, division
from shutil import copyfile
import csv
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
import torch
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets


with open("/mnt/Data/Downloads/data/coco/test2014.csv") as f:
    lines = f.read().splitlines()

for file in tqdm(lines):
    try:
        image = Image.open(f"/mnt/Data/Downloads/data/coco/test2014/{file}")
        assert (np.asarray(image).shape[2] == 3)
        image.save(f"/mnt/Data/Downloads/data/coco/test2014_sliced/{file}")
    except:
        pass

with open("/mnt/Data/Downloads/data/wikiart/test.csv") as f:
    lines = f.read().splitlines()

for file in tqdm(lines):
    try:
        image = Image.open(f"/mnt/Data/Downloads/data/wikiart/test/{file}")
        assert (np.asarray(image).shape[2] == 3)
        image.save(f"/mnt/Data/Downloads/data/wikiart/test_sliced/{file}")
    except:
        pass


# with open("/mnt/Data/Downloads/data/coco/train2014_sliced.csv") as f:
#     lines = f.read().splitlines()
#
# for file in tqdm(lines):
#     copyfile(f"/mnt/Data/Downloads/data/coco/train2014/{file}",
#     f"/mnt/Data/Downloads/data/coco/train2014_sliced/1/{file}")
#
# with open("/mnt/Data/Downloads/data/wikiart/train_sliced.csv") as f:
#     lines = f.read().splitlines()
#
# for file in tqdm(lines):
#     copyfile(f"/mnt/Data/Downloads/data/wikiart/train/{file}",
#     f"/mnt/Data/Downloads/data/wikiart/train_sliced/2/{file}")
#


# with open("/mnt/Data/Downloads/data/coco/train2014.csv") as f:
#     lines = f.read().splitlines()
#
# for file in tqdm(lines):
#     try:
#         image = Image.open(f"/mnt/Data/Downloads/data/coco/train2014/{file}")
#         assert (np.asarray(image).shape[2] == 3)
#         image.save(f"/mnt/Data/Downloads/data/coco/train2014_sliced/{file}")
#     except:
#         pass
#
# with open("/mnt/Data/Downloads/data/wikiart/train.csv") as f:
#     lines = f.read().splitlines()
#
# for file in tqdm(lines):
#     try:
#         image = Image.open(f"/mnt/Data/Downloads/data/wikiart/train/{file}")
#         assert (np.asarray(image).shape[2] == 3)
#         image.save(f"/mnt/Data/Downloads/data/wikiart/train_sliced/{file}")
#     except:
#         pass
