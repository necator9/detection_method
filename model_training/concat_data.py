import logging
import os
import pandas as pd
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)


def clean_by_margin(df_data_or, margin=5, img_res=(1280, 720)):
    df_data_p = df_data_or[(df_data_or.x > margin) & (df_data_or.x < img_res[0] - margin)
                           & (df_data_or.x + df_data_or.w > margin) & (
                                       df_data_or.x + df_data_or.w < img_res[0] - margin)
                           & (df_data_or.y > margin) & (df_data_or.y < img_res[1] - margin)
                           & (df_data_or.y + df_data_or.h > margin) & (
                                       df_data_or.y + df_data_or.h < img_res[1] - margin)]

    return df_data_p


def open_csv(path):
    path = os.path.join(sys.argv[1], path)
    logging.debug('Reading {}'.format(path))
    csv = pd.read_csv(path)
    csv['name'] = path
    csv.y_rw = csv.y_rw.round(1)

    logging.debug('Margin filtering {}'.format(path))

    return clean_by_margin(csv, img_res=(1280, 720))


# Pedestrian objects
woman_1 = open_csv('woman_1.csv')
# woman = open_csv('walking-woman-low-poly-1.obj.csv')
# woman = woman[woman.height <= 1.95]
# df_man = open_csv('standing_man.obj.csv')
# df_boy = open_csv('running-boy.obj.csv')
# df_walking_man = open_csv('walking-man.obj.csv')

# Fixes according to real data
boy_short = open_csv('short_man.csv')
thin_man = open_csv('thin_man.csv')

ped = pd.concat([boy_short, woman_1, thin_man], sort=False)
ped['o_class'] = np.ones(ped.shape[0])

# Group objects
pair_1 = open_csv('pair_1.csv')
pair_2 = open_csv('pair_2.csv')


df_high_woman = open_csv('walking-woman-low-poly-1.obj_short.csv')
df_high_woman = df_high_woman[df_high_woman.height > 2]
# df_high_man = open_csv('standing_man.obj_short.csv')
# df_high_man = df_high_man[df_high_man.height > 2]

# pair
short_group = open_csv('short_group.csv')
tall_group_1 = open_csv('tall_group_1.csv')
tall_group_2 = open_csv('tall_group_2.csv')


group = pd.concat([pair_1, pair_2, tall_group_1, tall_group_2, short_group], sort=False)
group['o_class'] = np.ones(group.shape[0]) * 2

# Cyclist objects
cyclist_1 = open_csv('cyclist_1.csv')
# cyclist_1 = cyclist_1[(cyclist_1.y_rotation > 80) & (cyclist_1.height > 1.6)]

df_cyclist_cut = open_csv('cyclist-1.obj.csv')
df_cyclist_cut = df_cyclist_cut[(df_cyclist_cut.y_rotation > 50) & (df_cyclist_cut.y_rotation < 80)
                                & (df_cyclist_cut.height > 1.6)]

cyclist = pd.concat([cyclist_1, df_cyclist_cut], sort=False)
cyclist['o_class'] = np.ones(cyclist.shape[0]) * 3

# Car objects
df_car = open_csv('car-3.obj.csv')
df_car = df_car[df_car.y_rotation > 50]

# Fixes according to real data
disp_car = open_csv('disp_car.csv')
car = pd.concat([df_car, disp_car], sort=False)

car['o_class'] = np.ones(car.shape[0]) * 4

# pd_all_data = pd.concat([ped, group, cyclist, car])
pd_all_data = pd.concat([ped, group, cyclist, car], sort=False)
logging.info('Data shape: {0}'.format(pd_all_data.shape))

# pd_all_data[['w_rw', 'h_rw', 'c_a_rw',  'd', 'angle', 'y_rw', 'o_class']].to_csv('all_data.csv')
sc1 = pd_all_data[(pd_all_data.angle == -16) & (pd_all_data.y_rw == -5)]
# sc2 = pd_all_data[(pd_all_data.angle == -21) & (pd_all_data.y_rw == -3)]
# sc3 = pd_all_data[(pd_all_data.angle == -13) & (pd_all_data.y_rw == -3.1)]

pd_all_data = pd.concat([sc1], sort=False) #  sc2, sc3
pd_all_data.to_csv('all_data.csv')

