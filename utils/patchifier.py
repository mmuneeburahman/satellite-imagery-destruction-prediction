import os
from PIL import Image
import numpy as np
from patchify import patchify, unpatchify
from tqdm import tqdm
from multiprocessing import Pool

def save_row_patches(patches_path, patches, row_idx):
    """
    patches_path: path where to save the image
    patches: N x H x W x C
    row_idx: index of row
    """
    for j in range(patches.shape[0]):
        single_patch_img = patches[j]
        im = Image.fromarray(single_patch_img)
        patch_name = patches_path + 'image_' + str(row_idx) + '_' + str(j) + ".png"
        im.save(patch_name)

class Patchifier():
  def __init__(self, patch_size, patches_path, unpatch_path, mask_path=None):
    self.patch_size = patch_size
    self.patches_path = patches_path
    self.unpatch_path = unpatch_path
    self.mask_path = mask_path
  
  def clear_patches(self):
    os.system("rm -r "+self.patches_path+"*.png")
  
  def set_rc(self, H, W):
    self.R, self.C = H//self.patch_size, W//self.patch_size
    
  
  def get_size(self, H, W):
    """
    return the perfect square Height and Width
    """
    self.set_rc(H, W)
    new_H = self.R*self.patch_size
    new_W = self.C*self.patch_size
    return new_H, new_W
  
  def set_config(self, image_shape):
    H, W, C = image_shape
    new_H, new_W = self.get_size(H, W)
    self.pi_shape = (new_H, new_W, C)
    self.lr_shape = (self.patch_size, new_W, C)
    self.lc_shape = (new_H, self.patch_size, C)
    self.br_shape = (self.patch_size, self.patch_size, C)
  
  def get_perfect_image(self, image, new_H, new_W):
    return image[:new_H, :new_W, :]
  
  def get_last_row(self, image, new_W):
    return image[-self.patch_size:, :new_W, :]
  
  def get_last_col(self, image, new_H):
    return image[:new_H, -self.patch_size:, :]

  def get_bottom_right_patch(self, image):
    return image[-self.patch_size:, -self.patch_size:, :]

  def get_all_parts(self, image):
    H, W, C = image.shape
    new_H, new_W = self.get_size(H, W)

    # get sections of image
    perfect_image = self.get_perfect_image(image, new_H, new_W)
    last_row = self.get_last_row(image, new_W)
    last_col = self.get_last_col(image, new_H)
    br_patch = self.get_bottom_right_patch(image)

    # set image shapes
    self.pi_shape = perfect_image.shape
    self.lr_shape = last_row.shape
    self.lc_shape = last_col.shape
    self.br_shape = br_patch.shape
    
    return perfect_image, last_row, last_col, br_patch
  
  def create_patches(self, perfect_image, last_row, last_col):
    pi_patches = patchify(perfect_image, (self.patch_size, self.patch_size, 3), step=self.patch_size)
    lr_patches = patchify(last_row, (self.patch_size, self.patch_size, 3), step=self.patch_size)
    lc_patches = patchify(last_col, (self.patch_size, self.patch_size, 3), step=self.patch_size)
    return pi_patches, lr_patches, lc_patches

  def save_pi_patches(self, pi_patches):
    """
    pi_patches: R, C, 1, H, W, C
    """
    print("saving_pi_patches")
    args = [(self.patches_path, pi_patches[i, :, 0], i) for i in range(pi_patches.shape[0])]
    with Pool(processes=2) as pool:
        _ = pool.starmap(save_row_patches, args)

  def save_lr_patches(self, lr_patches):
    """
    lr_patches: 1, C, 1, H, W, C
    """
    print("saving_lr_patches")
    for i in range(lr_patches.shape[1]):
      single_patch_img = lr_patches[0,i,0]
      im = Image.fromarray(single_patch_img)
      patch_name = self.patches_path + 'image_'+ 'R' + str(i)+ ".png"
      im.save(patch_name)

  def save_lc_patches(self, lc_patches):
    """
    lc_patches: R, 1, 1, H, W, C
    """
    print("saving_lc_patches")
    for i in range(lc_patches.shape[0]):
      single_patch_img = lc_patches[i,0,0]
      im = Image.fromarray(single_patch_img)
      patch_name = self.patches_path + 'image_'+ 'C' + str(i)+ ".png"
      im.save(patch_name)
  
  def save_br_patch(self, br_patch):
    print("saving_rc_patch")
    patch_name = self.patches_path + 'image_'+ 'RC' + ".png"
    im = Image.fromarray(br_patch)
    im.save(patch_name)

  def save_patches(self, pi_patches, lr_patches, lc_patches, br_patch):
    self.save_pi_patches(pi_patches)
    self.save_lr_patches(lr_patches)
    self.save_lc_patches(lc_patches)
    self.save_br_patch(br_patch)
  
  def patchify(self, image):
    perfect_image, last_row, last_col, br_patch = self.get_all_parts(image)
    print("perfect image (pi) shape: ", perfect_image.shape)
    print("last row (pi) shape: ", last_row.shape)
    print("last col (pi) shape: ", last_col.shape)
    print("bottom right image (pi) shape: ", br_patch.shape)
    pi_patches, lr_patches, lc_patches = self.create_patches(perfect_image, last_row, last_col)
    self.save_patches(pi_patches, lr_patches, lc_patches, br_patch)
  
  def load_patches(self, patches_path):
    print("loading_pi_patches")
    all_patches = []
    for i in tqdm(range(self.R)):
      row = []
      for j in range(self.C):
        image_name = patches_path + 'image_' + str(i) + '_' + str(j)+ ".png"
        img = Image.open(image_name)
        img = np.array(img,dtype=np.uint8)
        img = np.expand_dims(img, axis = 0)
        row.append(img)
      all_patches.append(row)
    return np.array(all_patches)
  
  def load_last_row(self, patches_path):
    print("loading_lr_patches")
    row = []
    for i in tqdm(range(self.C)):
      image_name = patches_path + 'image_'+ 'R' + str(i)+ ".png"
      img = Image.open(image_name)
      img = np.array(img,dtype=np.uint8)
      img = np.expand_dims(img, axis = 0)
      row.append(img)
    return np.expand_dims(np.array(row), axis=0)

  def load_last_col(self, patches_path):
    print("loading_lc_patches")
    col = []
    for i in tqdm(range(self.R)):
      image_name = patches_path + 'image_'+ 'C' + str(i)+ ".png"
      img = Image.open(image_name)
      img = np.array(img,dtype=np.uint8)
      img = np.expand_dims(np.expand_dims(img, axis = 0),axis=0)
      col.append(img)
    return np.array(col)

  def load_rc(self, patches_path):
    print("loading_rc_patch")
    image_RC = Image.open(patches_path+"image_RC.png")
    return np.array(image_RC,dtype=np.uint8)
  
  def unpatchify_separately(self):
    all_patches = self.load_patches(self.patches_path)
    llr = self.load_last_row(self.patches_path)
    llc = self.load_last_col(self.patches_path)

    #loading all the sections of tif image
    Image_RC = self.load_rc(self.patches_path)
    reconstructed_pi = unpatchify(all_patches, self.pi_shape)
    reconstructed_llr = unpatchify(llr, self.lr_shape)
    reconstructed_llc = unpatchify(llc, self.lc_shape)
    return reconstructed_pi, reconstructed_llr, reconstructed_llc, Image_RC

  def reconstruct_whole_image(self,extact_patch, last_row, last_col, image_RC, org_img_shape):
    H, W, _ = extact_patch.shape
    image = np.zeros((org_img_shape))
    image[:H, :W, :] = extact_patch
    image[-self.patch_size:, :W, :] = last_row
    image[:H, -self.patch_size:, :] = last_col
    image[-self.patch_size:, -self.patch_size:, :] = image_RC
    return image

  def unpatchify(self, org_img_shape):
    re_pi, re_llr, re_llc, image_rc = self.unpatchify_separately()
    return self.reconstruct_whole_image(re_pi, re_llr, re_llc, image_rc, org_img_shape)

  def unpatchify_separately_mask(self):
    all_patches = self.load_patches(self.mask_path)
    llr = self.load_last_row(self.mask_path)
    llc = self.load_last_col(self.mask_path)
    Image_RC = self.load_rc(self.mask_path)
    all_patches = np.expand_dims(all_patches, axis=-1)
    llr = np.expand_dims(llr, axis=-1)
    llc = np.expand_dims(llc, axis=-1)
    Image_RC = np.expand_dims(Image_RC, axis = 2)

    H, W, C = self.pi_shape
    reconstructed_pi = unpatchify(all_patches, (H, W, 1))

    H, W, C = self.lr_shape
    reconstructed_llr = unpatchify(llr, (H, W, 1))

    H, W, C = self.lc_shape
    reconstructed_llc = unpatchify(llc, (H, W, 1))
    return reconstructed_pi, reconstructed_llr, reconstructed_llc, Image_RC

  def reconstruct_whole_mask(self,extact_patch, last_row, last_col, image_RC, mask_shape):
    H, W, C = extact_patch.shape
    image = np.zeros((mask_shape), dtype=np.uint8)
    image[:H, :W, :] = extact_patch
    image[-self.patch_size:, :W, :] = last_row
    image[:H, -self.patch_size:, :] = last_col
    image[-self.patch_size:, -self.patch_size:, :] = image_RC
    return image.astype(np.uint8)


  def unpatchify_mask(self, mask_shape):
    re_pi, re_llr, re_llc, image_rc = self.unpatchify_separately_mask()
    return self.reconstruct_whole_image(re_pi, re_llr, re_llc, image_rc, mask_shape)
