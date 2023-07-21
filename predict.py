from subprocess import call
from os import listdir, path, makedirs
from utils.utils import unpatch_and_save, unpatch_and_save_mask

def predict_localization(pre_patches_dir):
  for tif_name in listdir(pre_patches_dir):
    print(f"predicting localization for {tif_name}")
    tif_patches_path = path.join(pre_patches_dir, tif_name)
    call(["python", "predict_loc.py", "1", tif_patches_path, tif_name])
    call(["python", "predict_loc.py", "2", tif_patches_path, tif_name])
    call(["python", "predict_loc.py", "3", tif_patches_path, tif_name])
    call(["python", "predict_loc.py", "4", tif_patches_path, tif_name])

def predict_cls(pre_patches_dir):
    for tif_name in listdir(pre_patches_dir):
        print(f"predicting destruction for {tif_name}")
        tif_patches_path = path.join(pre_patches_dir, tif_name)
        for seed in [0, 1, 2]:
            call(["python", "predict_cls.py", str(seed), "1", tif_patches_path, tif_name])
            call(["python", "predict_cls.py", str(seed), "2", tif_patches_path, tif_name])
            call(["python", "predict_cls.py", str(seed), "3", tif_patches_path, tif_name])
            call(["python", "predict_cls.py", str(seed), "4", tif_patches_path, tif_name])
    
def create_submission(pre_patches_dir):
    for tif_name in listdir(pre_patches_dir):
        print(f"predicting destruction for {tif_name}")
        call(["python", "create_submission.py", tif_name])

def create_overlays(pre_patches_dir):
    for tif_name in listdir(pre_patches_dir):
        print(f"Creating overlay for {tif_name}")
        call(["python", "create_overlay.py", tif_name])
        
def reconstruct(rgb_shape, tiff_type, re_type):
    makedirs("reconstruction", exist_ok=True)
    makedirs(path.join("reconstruction", tiff_type), exist_ok=True)
    
    des_overlay_dir = f"prediction/{tiff_type}/"

    print(f"Reconstructing destruction overlays...")
    for des_tif_name in listdir(des_overlay_dir):
        des_overlay_path = path.join(des_overlay_dir, des_tif_name)
        post_img_path = path.join("data/tiff/post", des_tif_name + ".tif")
        unpatch_and_save(
                        rgb_shape,
                        des_overlay_path,
                        post_img_path,
                        f"reconstruction/{tiff_type}/"+des_tif_name+f"-{re_type}.tif"
                        )

def reconstruct_mask(rgb_shape, data_dir, patches_dir):
    print(f"Reconstructing mask...")
    makedirs("reconstruction", exist_ok=True)
    makedirs(path.join("reconstruction", "localization"), exist_ok=True)
    makedirs(path.join("reconstruction", "destruction"), exist_ok=True)
    
    loc_msk_dir = f"prediction/submission/localization/"
    des_msk_dir = f"prediction/submission/destruction/"
    for tif_file in listdir(loc_msk_dir):
        pre_patches_path = path.join(patches_dir, "pre", tif_file, "")
        post_patches_path = path.join(patches_dir, "post", tif_file, "")

        des_pred_path = path.join(des_msk_dir, tif_file)
        loc_pred_path = path.join(loc_msk_dir, tif_file)

        pre_img_path = path.join(data_dir, "pre", tif_file+".tif")
        post_img_path = path.join(data_dir, "post", tif_file+".tif")

        unpatch_and_save_mask(rgb_shape,
                        post_patches_path,
                        des_pred_path,
                        post_img_path,
                        "reconstruction/destruction/"+tif_file+"-destruction-mask.tif"
                        )
        unpatch_and_save_mask(rgb_shape,
                        pre_patches_path,
                        loc_pred_path,
                        pre_img_path,
                        "reconstruction/localization/"+tif_file+"-localization-mask.tif"
                        )