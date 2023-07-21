import sys
import timeit
from multiprocessing import Pool
from os import path, makedirs, listdir

import numpy as np
from tqdm import tqdm

import imageio.v3 as iio
import rasterio.features
import shapely.geometry
from PIL import Image, ImageDraw

file_name = sys.argv[1]
pre_patches_path = "data/patches/pre"
local_dict = {
    "localization": (0, 255, 0, 100)
}
def create_localization_overlay(f):
    path_to_image = path.join("data/patches/pre", file_name, f)
    path_to_localization = path.join("prediction/submission/localization", file_name, f)
    img = Image.open(path_to_image)

    loc_arr = iio.imread(path_to_localization)
    #thresholding
    loc_arr = (loc_arr >= 1).astype(np.uint8)
    shapes = rasterio.features.shapes(loc_arr)

    localization_polygons = []
    for shape in shapes:
      if shape[1] == 1:
        localization_polygons.append(shapely.geometry.Polygon(shape[0]["coordinates"][0]))

    draw = ImageDraw.Draw(img, 'RGBA')

    # print(len(localization_polygons))
    # Go through each list and write it to the post image we just loaded
    for polygon in localization_polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, local_dict["localization"])

    path_to_save = path.join("prediction/localization_overlay", file_name, f)
    img.save(path_to_save)

def create_destruction_overlay(f):
    path_to_image = path.join("data/patches/pre", file_name, f)
    path_to_localization = path.join("prediction/submission/localization", file_name, f)
    path_to_damage = path.join("prediction/submission/destruction", file_name, f)
    path_to_output = path.join("prediction/destruction_overlay", file_name, f)

    no_damage_polygons = []
    minor_damage_polygons = []
    major_damage_polygons = []
    destroyed_polygons = []

    # Load the challenge output localization image
    localization = Image.open(path_to_localization)
    loc_arr = np.array(localization)

    # If the localization has damage values convert all non-zero to 1
    # This helps us find where buildings are, and then use the damage file
    # to get the value of the classified damage
    loc_arr = (loc_arr >= 1).astype(np.uint8)

    # Load the challenge output damage image
    damage = Image.open(path_to_damage)
    dmg_arr = np.array(damage)

    # Use the localization to get damage only were they have detected buildings
    mask_arr = dmg_arr*loc_arr

    # Get the value of each index put into a dictionary like structure
    shapes = rasterio.features.shapes(mask_arr)

    # Iterate through the unique values of the shape files
    # This is a destructive iterator or else we'd use the pythonic for x in shapes if x blah
    for shape in shapes:
        if shape[1] == 1:
            no_damage_polygons.append(shapely.geometry.Polygon(shape[0]["coordinates"][0]))
        elif shape[1] == 2:
            minor_damage_polygons.append(shapely.geometry.Polygon(shape[0]["coordinates"][0]))
        elif shape[1] == 3:
            major_damage_polygons.append(shapely.geometry.Polygon(shape[0]["coordinates"][0]))
        elif shape[1] == 4:
            destroyed_polygons.append(shapely.geometry.Polygon(shape[0]["coordinates"][0]))
        elif shape[1] == 0:
            continue
        else:
            print("Found non-conforming damage type: {}".format(shape[1]))

    # Loading post image
    img = Image.open(path_to_image)

    draw = ImageDraw.Draw(img, 'RGBA')

    damage_dict = {
        "no-damage": (0, 255, 0, 100), #greenish
        "minor-damage": (0, 0, 255, 125), #blueish
        "major-damage": (255, 155, 0, 125), #orange
        "destroyed": (255, 0, 0, 125),  #red
        "un-classified": (255, 255, 255, 125)
    }
    # Go through each list and write it to the post image we just loaded
    for polygon in no_damage_polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict["no-damage"])

    for polygon in minor_damage_polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict["minor-damage"])

    for polygon in major_damage_polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict["major-damage"])

    for polygon in destroyed_polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict["destroyed"])

    img.save(path_to_output)

if __name__ == '__main__':
    t0 = timeit.default_timer()
    sub_folder = "prediction/localization_overlay"
    makedirs(sub_folder, exist_ok=True)
    makedirs(path.join(sub_folder, file_name), exist_ok=True)

    # reading names of all files.
    all_files = []
    for f in tqdm(sorted(listdir(path.join(pre_patches_path, file_name)))):
        all_files.append(f)


    with Pool() as pool:
        _ = pool.map(create_localization_overlay, all_files)

    sub_folder = "prediction/destruction_overlay"
    makedirs(sub_folder, exist_ok=True)
    makedirs(path.join(sub_folder, file_name), exist_ok=True)
    with Pool() as pool:
        _ = pool.map(create_destruction_overlay, all_files)


    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))