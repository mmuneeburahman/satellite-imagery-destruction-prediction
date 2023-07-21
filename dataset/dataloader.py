import random
from os import path, makedirs, listdir

import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
from imgaug import augmenters as iaa

from dataset.augmentation import *

input_shape = (736, 736)
train_dirs = ['train', 'tier3']

all_files = [] #contain _pre_disaster image names in train and tier3 folder.
for d in train_dirs:
    for f in sorted(listdir(path.join(d, 'images'))):
        if '_pre_disaster.png' in f:
            all_files.append(path.join(d, 'images', f))


class TrainData(Dataset):
    def __init__(self, train_idxs):
        super().__init__()
        self.train_idxs = train_idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]

        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        
        if random.random() > 0.985:
            img = cv2.imread(fn.replace('_pre_disaster', '_post_disaster'), cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)

        if random.random() > 0.5:
            # FLIP_TOP_BOTTOM
            img = img[::-1, ...]
            msk0 = msk0[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4) #generate a number from [0,4) and rotate 90 deg rot times.
            if rot > 0:
                img = np.rot90(img, k=rot)
                msk0 = np.rot90(msk0, k=rot)

        if random.random() > 0.8:
            #shift 320 pixel up or down, and 320 pixel left or right.
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)
            msk0 = shift_image(msk0, shift_pnt)
            
        if random.random() > 0.2:
            # rotate and scale image.
            rot_pnt =  (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)
                msk0 = rotate_image(msk0, angle, scale, rot_pnt)

        crop_size = input_shape[0]
        # get crop values for n times, where n = random number in range(1, 5).
        # retain the crop with maximum localization in mask.
        if random.random() > 0.3:
            crop_size = random.randint(int(input_shape[0] / 1.2), int(input_shape[0] / 0.8))

        bst_x0 = random.randint(0, img.shape[1] - crop_size)
        bst_y0 = random.randint(0, img.shape[0] - crop_size)
        bst_sc = -1
        try_cnt = random.randint(1, 5)
        for i in range(try_cnt):
            x0 = random.randint(0, img.shape[1] - crop_size)
            y0 = random.randint(0, img.shape[0] - crop_size)
            _sc = msk0[y0:y0+crop_size, x0:x0+crop_size].sum()
            if _sc > bst_sc:
                bst_sc = _sc
                bst_x0 = x0
                bst_y0 = y0
        x0 = bst_x0
        y0 = bst_y0
        img = img[y0:y0+crop_size, x0:x0+crop_size, :]
        msk0 = msk0[y0:y0+crop_size, x0:x0+crop_size]

        
        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)
            msk0 = cv2.resize(msk0, input_shape, interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.97:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))
        elif random.random() > 0.97:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.93:
            if random.random() > 0.97:
                img = clahe(img)
            elif random.random() > 0.97:
                img = gauss_noise(img)
            elif random.random() > 0.97:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.93:
            if random.random() > 0.97:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.97:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.97:
                img = contrast(img, 0.9 + random.random() * 0.2)
                
        if random.random() > 0.97:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        msk = msk0[..., np.newaxis]

        msk = (msk > 127) * 1

        img = preprocess_inputs(img)

        #numpy.ndarray (H x W x C) in the range
        #[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'fn': fn}
        return sample


class ValData(Dataset):
    def __init__(self, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]

        fn = all_files[_idx]

        img = cv2.imread(fn, cv2.IMREAD_COLOR)

        msk0 = cv2.imread(fn.replace('/images/', '/masks/'), cv2.IMREAD_UNCHANGED)

        msk = msk0[..., np.newaxis]

        msk = (msk > 127) * 1

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        msk = torch.from_numpy(msk.transpose((2, 0, 1))).long()

        sample = {'img': img, 'msk': msk, 'fn': fn}
        return sample
