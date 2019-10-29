import pandas as pd
import numpy as np
import pickle
import os
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    path = os.path.join('csv_dir', path)
    logging.debug('Reading {}'.format(path))
    csv = pd.read_csv(path) # , index_col=0
    logging.debug('Margin filtering {}'.format(path))
    try:
        csv = csv[csv.thr == 70]
    except AttributeError:
        logging.info('df {} does not have thr'.format(path))

    return clean_by_margin(csv, img_res=(424, 240))


def augment_ca(n_df):
    ca_coef = np.linspace(0.1, 1, 10)
    n_df.c_a_rw = (n_df.c_a_px * (n_df.w_rw * n_df.h_rw)) / (n_df.w * n_df.h)

    for ca_c in ca_coef:
        n_df[str(ca_c) + 'ca_coef'] = n_df.c_a_rw * ca_c

    n_df = n_df.melt(id_vars=['d', 'c_a_rw', 'w_rw', 'h_rw', 'extent', 'x', 'y', 'w', 'h', 'c_a_px', 'x_rw',
                              'y_rw', 'z_rw', 'y_rotation', 'width', 'height', 'depth', 'angle'],
                     var_name="ca_coef", value_name="ca_aug")

    n_df.c_a_rw = n_df.ca_aug

    return n_df


# Pedestrian objects
df_woman = open_csv('walking-woman-low-poly-1.obj.csv')
df_boy = open_csv('running-boy.obj.csv')
df_walking_man = open_csv('walking-man.obj.csv')
df_st_man = open_csv('stepping_man.obj.csv')
df_ped = pd.concat([df_woman, df_boy, df_walking_man, df_st_man], sort=False)

# Group objects
#df_pair = open_csv('pair.obj.csv')
df_group_1 = open_csv('group-1.obj.csv')
df_group_2 = open_csv('group-2.obj.csv')
df_group = pd.concat([df_group_1, df_group_2], sort=False) # df_pair,
# df_pair = df_pair[df_pair.y_rotation > 80]

# logging.info(df_pair.shape)

# Cyclist objects
# df_cyclist = open_csv('cyclist-1.obj.csv')


# Noise objects
# df_noise = open_csv('noises_500_cycle.csv')
# df_noise = open_csv('noises_500.csv')
# df_noise = augment_ca(open_csv('noises_500_ped_pair.csv'))
# df_noise = open_csv('noises_500_ped_pair.csv')
df_noise = open_csv('noises_500_ped_pair_new.csv')
# df_noise = open_csv('noises_500_ped.csv')


noise = np.vstack((df_noise.w_rw, df_noise.h_rw, df_noise.c_a_rw, df_noise.y_rw, df_noise.d, df_noise.angle,
                   np.zeros(df_noise.shape[0]))).T
ped = np.vstack((df_ped.w_rw, df_ped.h_rw, df_ped.c_a_rw, df_ped.y_rw, df_ped.d, df_ped.angle,
                 np.ones(df_ped.shape[0]))).T
pair = np.vstack((df_group.w_rw, df_group.h_rw, df_group.c_a_rw, df_group.y_rw, df_group.d,
                  df_group.angle, np.ones(df_group.shape[0]) * 2)).T
# cyclist = np.vstack((df_cyclist.w_rw, df_cyclist.h_rw, df_cyclist.c_a_rw, df_cyclist.y_rw, df_cyclist.d,
#                      df_cyclist.angle, np.ones(df_cyclist.shape[0]) * 3)).T


training_data = np.concatenate((noise, ped, pair))  # , ,cyclist cyclist)
pd_all_data = pd.DataFrame(training_data, columns=['w_rw', 'h_rw', 'c_a_rw', 'y_rw', 'd', 'angle', 'o_class'])
logging.info('Data shape: {0}'.format(pd_all_data.shape))

features_cols = [0, 1, 3, 4, 5] # range(0, 6)
X_ = training_data[:, features_cols]
y_ = training_data[:, -1]

# Scale the data
scaler = StandardScaler()
X_ = scaler.fit_transform(X_)

poly = PolynomialFeatures(2, include_bias=True)
X_, y_ = poly.fit_transform(X_), y_

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=.3, random_state=42)

clf = LogisticRegression(solver='newton-cg', C=3, multi_class='auto', n_jobs=-1, max_iter=1000)  #, C=5, max_iter=100
clf.fit(X_train, y_train)

human_scaled = scaler.transform(ped[:, features_cols])
pair_scaled = scaler.transform(pair[:, features_cols])
# cyclist_scaled = scaler.transform(cyclist[:, features_cols])

logging.info('overall score {}'.format(clf.score(X_test, y_test)))
logging.info('ped acc {}'.format(clf.score(poly.fit_transform(human_scaled), ped[:, -1])))
logging.info('group acc {}'.format(clf.score(poly.fit_transform(pair_scaled), pair[:, -1])))
# logging.info('cyclist acc {}'.format(clf.score(poly.fit_transform(cyclist_scaled), cyclist[:, -1])))

with open('clf/ped_pair_clf.pcl', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('clf/ped_pair_scale.pcl', 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
