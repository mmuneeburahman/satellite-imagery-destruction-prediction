from os import path, makedirs
import hydra
from omegaconf.dictconfig import DictConfig

from utils.utils import generate_patches
from predict import *

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    task = cfg.task
    pre_patches_dir = path.join(cfg.patches_dir, "pre")
    post_patches_dir = path.join(cfg.patches_dir, "post")

    if task.generate_patches:
        makedirs(pre_patches_dir, exist_ok=True)
        generate_patches(path.join(cfg.data_dir, "pre"), path.join(cfg.patches_dir, "pre"))
        
        makedirs(post_patches_dir, exist_ok=True)
        generate_patches(path.join(cfg.data_dir, "post"), path.join(cfg.patches_dir, "post"))
    
    makedirs(cfg.prediction_dir, exist_ok=True)
    if task.predict_loc:
        predict_localization(pre_patches_dir)

    if task.predict_cls:
        predict_cls(pre_patches_dir)
    
    if task.create_sub:
        create_submission(pre_patches_dir)
    
    if task.create_overlay:
        create_overlays(pre_patches_dir)
    
    if task.reconstruct_overlay:
        print(type(cfg.rgb_shape))
        reconstruct(cfg.rgb_shape, "localization_overlay", "localization")
        reconstruct(cfg.rgb_shape, "destruction_overlay", "destruction")
    
    if task.reconstruct_mask:
        reconstruct_mask(cfg.rgb_shape, cfg.data_dir, cfg.patches_dir)
        
if __name__ == "__main__":
    main()