import pandas as pd
import numpy as np
import pickle
import logging
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score


logging.basicConfig(level=logging.INFO)

# Target group
target = pd.read_csv(sys.argv[1] + '/all_data.csv')

# Noise objects
df_noise = pd.read_csv(sys.argv[1] + '/new_noises_3000.csv')

pd_all_data = pd.concat([target, df_noise], sort=False)
logging.info('Data shape: {0}'.format(pd_all_data.shape))

training_data = np.vstack((pd_all_data.w_rw, pd_all_data.h_rw, pd_all_data.c_a_rw, pd_all_data.d,
                           pd_all_data.y_rw, pd_all_data.angle, pd_all_data.o_class)).T

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
logging.info('Precision - P=TP/TP+FP: {}'.format(precision_score(y_test, clf.predict(X_test), average=None)))
logging.info('Recall - R=TP/TP+FN: {} '.format(recall_score(y_test, clf.predict(X_test), average=None)))
logging.info('F1 score - F1=2*(P*R)/(P+R): {}\n'.format(f1_score(y_test, clf.predict(X_test), average=None)))
logging.info(confusion_matrix(y_test, clf.predict(X_test), labels=[0, 1, 2, 3, 4]))

with open('clf/clf_3.pcl', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('clf/scaler_3.pcl', 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
