import rioxarray as rio 
from .patchifier import Patchifier
from os import listdir, path, makedirs
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def iou(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    union = np.logical_or(im1, im2)
    im_sum = union.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return intersection.sum() / im_sum



## MyUtil Functions
def generate_patches(tif_files_dir, output_dir):
  """
  tif_files_dir: path to directory that contains tif files
  output_dir: path to directory that store the patches

  This function reads tif files from the dir and create a directory
  same name as tif in the output directory and store the patches
  """
  for tif in listdir(tif_files_dir):
    # Create directory with tif filename
    output_patches_path = path.join(output_dir, tif.split('.')[0])
    makedirs(output_patches_path, exist_ok=True)

    # read tif file
    pre_img_path = path.join(tif_files_dir, tif)
    pre_img = rio.open_rasterio(pre_img_path)

    #initialize patchifier
    pfr = Patchifier(1024, output_patches_path+"/", None)

    # changing dimension to convert to rgb
    pfr.patchify(pre_img.data.transpose((1,2, 0)))


def unpatch_and_save(rgb_shape, patches_path, tif_image_path, path_to_save):
  pfr = Patchifier(1024, patches_path+"/", None)
  pfr.set_config(rgb_shape)
  re_image = pfr.unpatchify(rgb_shape)

  img = rio.open_rasterio(tif_image_path)
  img.data = np.uint8(re_image.transpose((2, 0, 1)))
  
  del re_image
  img.rio.to_raster(raster_path=path_to_save)

def unpatch_and_save_mask(rgb_shape, patches_path, mask_path, tif_image_path, path_to_save):
  pfr = Patchifier(1024, patches_path+"/", None, mask_path+"/")
  pfr.set_config(rgb_shape)
  H, W, C = rgb_shape
  re_mask = pfr.unpatchify_mask((H, W, 1))

  img = rio.open_rasterio(tif_image_path)
  img = img.drop(2, dim='band')
  img = img.drop(3, dim='band')

  img.data = np.uint8(re_mask.transpose((2, 0, 1)))
  del re_mask


  img.rio.to_raster(raster_path=path_to_save, compress='zstd')