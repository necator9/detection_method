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
    csv.y_rw = csv.y_rw.round(1)

    logging.debug('Margin filtering {}'.format(path))

    return clean_by_margin(csv, img_res=(424, 240))


# Pedestrian objects
df_woman = open_csv('walking-woman-low-poly-1.obj.csv')
df_boy = open_csv('running-boy.obj.csv')
df_walking_man = open_csv('walking-man.obj.csv')
ped = pd.concat([df_woman, df_boy, df_walking_man], sort=False)
ped['o_class'] = np.ones(ped.shape[0])

# Group objects
df_pair = open_csv('pair.obj.csv')
group = pd.concat([df_pair], sort=False)
group['o_class'] = np.ones(group.shape[0]) * 2

# Cyclist objects
df_cyclist = open_csv('cyclist-1.obj.csv')
cyclist = pd.concat([df_cyclist], sort=False)
cyclist = cyclist[cyclist.y_rotation > 50]
cyclist['o_class'] = np.ones(cyclist.shape[0]) * 3

# # Car objects
# df_car = open_csv('car-3.obj.csv')
# car = pd.concat([df_car], sort=False)
# car = car[car.y_rotation > 22]
# car['o_class'] = np.ones(car.shape[0]) * 4

pd_all_data = pd.concat([ped, group, cyclist])  # , car
logging.info('Data shape: {0}'.format(pd_all_data.shape))

pd_all_data[['w_rw', 'h_rw', 'c_a_rw',  'd', 'angle', 'y_rw', 'o_class']].to_csv('all_data.csv')
