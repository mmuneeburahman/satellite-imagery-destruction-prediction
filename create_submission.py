import os

from os import path, makedirs, listdir
import sys
sys.setrecursionlimit(10000)
from multiprocessing import Pool
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from skimage.morphology import remove_small_objects

import matplotlib.pyplot as plt
import seaborn as sns

from skimage.morphology import square, dilation

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# test_dir = 'test/images'
file_name = sys.argv[1]
pred_folders = ['prediction/destruction/dpn92cls_cce_0_tuned', 'prediction/destruction/dpn92cls_cce_1_tuned', 'prediction/destruction/dpn92cls_cce_2_tuned'] + \
                ['prediction/destruction/res34cls2_0_tuned', 'prediction/destruction/res34cls2_1_tuned', 'prediction/destruction/res34cls2_2_tuned'] + \
                ['prediction/destruction/res50cls_cce_0_tuned', 'prediction/destruction/res50cls_cce_1_tuned', 'prediction/destruction/res50cls_cce_2_tuned'] + \
                ['prediction/destruction/se154cls_0_tuned', 'prediction/destruction/se154cls_1_tuned', 'prediction/destruction/se154cls_2_tuned']

pred_coefs = [1.0] * 12
loc_folders = ['prediction/localization/pred50_loc_tuned', 
               'prediction/localization/pred92_loc_tuned', 
               'prediction/localization/pred34_loc', 
               'prediction/localization/pred154_loc']
loc_coefs = [1.0] * 4 

sub_folder = 'prediction/submission'
sub_folder_loc = path.join(sub_folder, "localization")
sub_folder_des = path.join(sub_folder, "destruction")

_thr = [0.38, 0.13, 0.14]

def process_image(f):
    preds = []
    _i = -1
    for d in pred_folders:
        _i += 1
        msk1 = cv2.imread(path.join(d, file_name, f.replace('.png', '_part1.png.png')), cv2.IMREAD_UNCHANGED)
        msk2 = cv2.imread(path.join(d, file_name, f.replace('.png', '_part2.png.png')), cv2.IMREAD_UNCHANGED)
        msk = np.concatenate([msk1, msk2[..., 1:]], axis=2)
        preds.append(msk * pred_coefs[_i])
    preds = np.asarray(preds).astype('float').sum(axis=0) / np.sum(pred_coefs) / 255
    
    loc_preds = []
    _i = -1
    for d in loc_folders:
        _i += 1
        msk = cv2.imread(path.join(d, file_name, f), cv2.IMREAD_UNCHANGED)
        loc_preds.append(msk * loc_coefs[_i])
    loc_preds = np.asarray(loc_preds).astype('float').sum(axis=0) / np.sum(loc_coefs) / 255

    loc_preds = loc_preds 

    msk_dmg = preds[..., 1:].argmax(axis=2) + 1
    msk_loc = (1 * ((loc_preds > _thr[0]) | ((loc_preds > _thr[1]) & (msk_dmg > 1) & (msk_dmg < 4)) | ((loc_preds > _thr[2]) & (msk_dmg > 1)))).astype('uint8')
    
    msk_dmg = msk_dmg * msk_loc
    _msk = (msk_dmg == 2)
    if _msk.sum() > 0:
        _msk = dilation(_msk, square(5))
        msk_dmg[_msk & msk_dmg == 1] = 2

    msk_dmg = msk_dmg.astype('uint8')
    cv2.imwrite(path.join(sub_folder_loc, file_name, f), msk_loc, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(path.join(sub_folder_des, file_name, f), msk_dmg, [cv2.IMWRITE_PNG_COMPRESSION, 9])


if __name__ == '__main__':
    t0 = timeit.default_timer()

    makedirs(sub_folder, exist_ok=True)
    makedirs(sub_folder_loc, exist_ok=True)
    makedirs(sub_folder_des, exist_ok=True)
    makedirs(path.join(sub_folder_loc, file_name), exist_ok=True)
    makedirs(path.join(sub_folder_des, file_name), exist_ok=True)

    all_files = []
    files_path = path.join(loc_folders[0], file_name)
    
    for f in tqdm(sorted(listdir(files_path))):
        # if '_part1.png' in f:
            all_files.append(f)
    # print(all_files[0])

    with Pool() as pool:
        _ = pool.map(process_image, all_files)

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))