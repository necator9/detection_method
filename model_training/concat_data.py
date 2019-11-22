import logging
import os
import pandas as pd
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

    logging.debug('Margin filtering {}'.format(path))

    return clean_by_margin(csv, img_res=(1280, 720))


# Pedestrian objects
woman_1 = open_csv('woman_1.csv')
# Fixes according to real data
boy_short = open_csv('short_man.csv')
thin_man = open_csv('thin_man.csv')
ped = pd.concat([woman_1, boy_short, thin_man], sort=False)

# Group objects
pair_1 = open_csv('pair_1.csv')
pair_2 = open_csv('pair_2.csv')
# df_high_man = open_csv('standing_man.obj_short.csv')
# df_high_man = df_high_man[df_high_man.height > 2]
short_group = open_csv('short_group.csv')
tall_group_1 = open_csv('tall_group_1.csv')
tall_group_2 = open_csv('tall_group_2.csv')
group = pd.concat([pair_1, pair_2, tall_group_1, tall_group_2, short_group], sort=False)

# Cyclist objects
cyclist_1 = open_csv('cyclist_1.csv')

# Car objects
car_1 = open_csv('car_1.csv')

# Fixes according to real data
disp_car = open_csv('disp_car.csv')
car = pd.concat([disp_car, car_1], sort=False)

data = pd.concat([ped, group, cyclist_1, car], sort=False)
logging.info('Data shape: {0}'.format(data.shape))

# pd_all_data[['w_rw', 'h_rw', 'c_a_rw',  'd', 'angle', 'y_rw', 'o_class']].to_csv('all_data.csv')
# sc1 = pd_all_data[(pd_all_data.angle == -16) & (pd_all_data.y_rw == -5)]
# sc2 = pd_all_data[(pd_all_data.angle == -21) & (pd_all_data.y_rw == -3)]
# sc3 = pd_all_data[(pd_all_data.angle == -13) & (pd_all_data.y_rw == -3.1)]

# pd_all_data = pd.concat([sc1], sort=False) #  sc2, sc3
data = data.astype({'o_class': int})
data.to_csv(os.path.join(sys.argv[1], 'all_data.csv'))

