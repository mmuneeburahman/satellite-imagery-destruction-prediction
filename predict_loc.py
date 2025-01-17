import os

from os import path, makedirs, listdir
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

import pandas as pd
from tqdm import tqdm
import timeit
import cv2

from zoo.models import Res34_Unet_Loc, SeResNext50_Unet_Loc
from zoo.models import Dpn92_Unet_Loc, SeNet154_Unet_Loc

# from utils import *
from dataset.augmentation import preprocess_inputs

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

test_dir = sys.argv[2]
file_name = sys.argv[3]
models_folder = 'weights'


if __name__ == '__main__':
  t0 = timeit.default_timer()

  
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

  # seed = int(sys.argv[1])
  # seeds 
  models = []
  for seed in [0, 1, 2]:
    model_num = int(sys.argv[1])
    if model_num == 1:
      pred_folder = 'pred34_loc'
      snap_to_load = 'res34_loc_{}_1_best'.format(seed)
      model = Res34_Unet_Loc().cuda()
      model = nn.DataParallel(model).cuda()
    elif model_num == 2:
      pred_folder = 'pred50_loc_tuned'
      snap_to_load = 'res50_loc_{}_tuned_best'.format(seed)
      model = SeResNext50_Unet_Loc().cuda()
    elif model_num == 3:
      pred_folder = 'pred92_loc_tuned'
      snap_to_load = 'dpn92_loc_{}_tuned_best'.format(seed)
      model = Dpn92_Unet_Loc().cuda()
    elif model_num == 4:
      pred_folder = 'pred154_loc'
      snap_to_load = 'se154_loc_{}_1_best'.format(seed)
      model = SeNet154_Unet_Loc().cuda()
      model = nn.DataParallel(model).cuda()

      
    print("=> loading checkpoint '{}'".format(snap_to_load))
    checkpoint = torch.load(path.join(models_folder, snap_to_load), map_location='cpu')
    loaded_dict = checkpoint['state_dict']
    sd = model.state_dict()
    for k in model.state_dict():
      if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
        sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print("loaded checkpoint '{}' (epoch {}, best_score {})"
        .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
    model.eval()
    models.append(model)

  pred_folder = path.join("prediction/localization", pred_folder, file_name)
  makedirs(pred_folder, exist_ok=True)


  with torch.no_grad():
    for f in tqdm(sorted(listdir(test_dir))):
      # if '_pre_' in f:
        fn = path.join(test_dir, f)

        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        img = preprocess_inputs(img)

        inp = []
        inp.append(img)
        inp.append(img[::-1, ...])
        inp.append(img[:, ::-1, ...])
        inp.append(img[::-1, ::-1, ...])
        inp = np.asarray(inp, dtype='float')
        inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
        inp = Variable(inp).cuda()

        pred = []
        for model in models:
          msk = model(inp)
          msk = torch.sigmoid(msk)
          msk = msk.cpu().numpy()
          
          pred.append(msk[0, ...])
          pred.append(msk[1, :, ::-1, :])
          pred.append(msk[2, :, :, ::-1])
          pred.append(msk[3, :, ::-1, ::-1])

        # Average prediction with nodel seed 0, 1, 2
        pred_full = np.asarray(pred).mean(axis=0) 
        
        msk = pred_full * 255
        msk = msk.astype('uint8').transpose(1, 2, 0)
        cv2.imwrite(path.join(pred_folder, f), msk[..., 0], [cv2.IMWRITE_PNG_COMPRESSION, 9])

  elapsed = timeit.default_timer() - t0
  print('Time: {:.3f} min'.format(elapsed / 60))