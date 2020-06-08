# Created by Ivan Matveev at 20.05.20
# E-mail: ivan.matveev@hs-anhalt.de

import glob
import os
import shutil


def check_if_dir_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)


dirs_to_merge = ['/mnt/data_partition/experiments/sources/TZK_scene_1/5m_ped_filtered_renamed/',
                 '/mnt/data_partition/experiments/sources/TZK_scene_1/11m_ped_filtered_renamed/',
                 '/mnt/data_partition/experiments/sources/TZK_scene_1/16m_ped_filtered_renamed/',
                 '/mnt/data_partition/experiments/sources/TZK_scene_1/bicyclist_random_filtered/',
                 '/mnt/data_partition/experiments/sources/TZK_scene_1/chaos/']

out_path = '/mnt/data_partition/experiments/sources/TZK_scene_1/all_merged/'

check_if_dir_exists(out_path)

all_img_names_sorted = list()
for in_dir in dirs_to_merge:
    img_paths = glob.glob(os.path.join(in_dir, '*.jpeg'))
    # Extract digits only from an image name
    img_names_digits = [int(''.join([dig for dig in os.path.split(img_name)[1] if dig.isdigit()]))
                        for img_name in img_paths]
    img_names_digits, img_paths = zip(*sorted(zip(img_names_digits, img_paths)))
    all_img_names_sorted += img_paths

# Copy files from all in directories into a single one and numerate them ascendingly
for i, old_path in enumerate(all_img_names_sorted):
    shutil.copyfile(old_path, os.path.join(out_path, '{0}.jpeg'.format(str(i).zfill(4))))




