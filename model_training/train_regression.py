import pandas as pd
import numpy as np
import pickle
import os
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, average_precision_score, confusion_matrix, f1_score


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
    path = os.path.join('new_csv', path)
    logging.debug('Reading {}'.format(path))
    csv = pd.read_csv(path) # , index_col=0
    logging.debug('Margin filtering {}'.format(path))
    # try:
    #     csv = csv[csv.thr == 70]
    # except AttributeError:
    #     logging.info('df {} does not have thr'.format(path))

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

# Car objects
df_car = open_csv('car-3.obj.csv')
car = pd.concat([df_car])
car = car[car.y_rotation > 22]
car['o_class'] = np.ones(car.shape[0]) * 4

# Noise objects
df_noise = open_csv('noises_3000_all_x0.csv')
mu, sigma = 0.5, 0.1
df_noise.c_a_rw = np.random.normal(mu, sigma, df_noise.shape[0]) * (df_noise.w_rw * df_noise.h_rw)
df_noise['o_class'] = np.zeros(df_noise.shape[0])

pd_all_data = pd.concat([ped, group, cyclist, car, df_noise])
logging.info('Data shape: {0}'.format(pd_all_data.shape))

training_data = np.vstack((pd_all_data.w_rw, pd_all_data.h_rw, pd_all_data.c_a_rw, pd_all_data.d,
                           pd_all_data.angle, pd_all_data.y_rw, pd_all_data.o_class)).T

features_cols = range(6)  # [0, 1, 3, 4, 5]
X_ = training_data[:, features_cols]
y_ = training_data[:, -1]

# Scale the data
scaler = StandardScaler()
X_ = scaler.fit_transform(X_)

poly = PolynomialFeatures(2, include_bias=True)
X_, y_ = poly.fit_transform(X_), y_

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=.3, random_state=42)

clf = LogisticRegression(solver='newton-cg', C=3, multi_class='auto', n_jobs=-1, max_iter=100)
clf.fit(X_train, y_train)

logging.info('Accuracy - TP+TN/TP+FP+FN+TN: {}'.format(clf.score(X_test, y_test)))
logging.info('Presision - P=TP/TP+FP: {}'.format(precision_score(y_test, clf.predict(X_test), average=None)))
logging.info('Recall - R=TP/TP+FN: {} '.format(recall_score(y_test, clf.predict(X_test), average=None)))
logging.info('F1 score - F1=2*(P*R)/(P+R): {}\n'.format(f1_score(y_test, clf.predict(X_test), average=None)))
logging.info(confusion_matrix(y_test, clf.predict(X_test), labels=[0, 1, 2, 3, 4]))

with open('clf/clf.pcl', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('clf/scaler.pcl', 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
