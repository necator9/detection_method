from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import itertools
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def get_hull(cnt_data):
    hull = ConvexHull(cnt_data)
    hull = cnt_data[hull.vertices, :]
    hull = Delaunay(hull)

    return hull


def in_hull(p, hull):
    return hull.find_simplex(p) >= 0


def sel_data(df, cls):
    df = df[df.o_class == cls]
    df = np.vstack([df.w_rw, df.h_rw]).T

    return df


def find_rg(rg, margin=1.5):
    def check_val(val):
        return 0.01 if val <= 0 else val

    res_rg = check_val(min(rg) - margin), max(rg) + margin

    return res_rg


def gen_w_h(hulls_, points_amount_, w_rg_, h_rg_):
    noises = []
    while True:
        point = [np.random.uniform(*w_rg_), np.random.uniform(*h_rg_)]
        res_ = [in_hull(point, hull) for hull in hulls_]

        if not (any(res_)):
            noises.append(point)
            if len(noises) >= points_amount_:
                return np.array(noises)


points_amount = 3000

pd_all_data = pd.read_csv('new_csv/all_data.csv')
pd_all_data.y_rw = pd_all_data.y_rw.round(1)

angle_rg = np.arange(0, -70, -5)
height_rg = np.arange(-2, -7, -0.2)

lens = len(angle_rg) * len(height_rg)
it_params = itertools.product(angle_rg, height_rg)

noise = pd.DataFrame(columns=['w_rw', 'h_rw', 'c_a_rw', 'd', 'angle', 'y_rw', 'o_class'])

it = 0
for angle, height in it_params:
    try:
        a_h_data = pd_all_data[(pd_all_data.o_class != 0) & (pd_all_data.angle == angle)
                               & (pd_all_data.y_rw == round(height, 1))]
        w_rg = find_rg((a_h_data.w_rw.min(), a_h_data.w_rw.max()))
        h_rg = find_rg((a_h_data.h_rw.min(), a_h_data.h_rw.max()))

        hulls = []
        for i in range(1, 5):
            try:
                hulls.append(get_hull(sel_data(a_h_data, i)))

            except Exception:
                continue

        if len(hulls) == 0:
            continue

        w_h = gen_w_h(hulls, points_amount, w_rg, h_rg)

        d_rg = find_rg((a_h_data.d.min(), a_h_data.d.max()), margin=0.5)
        d = np.expand_dims(np.array([round(i) for i in np.random.uniform(*d_rg, size=[points_amount, 1])]), axis=1)

        mu, sigma = 0.5, 0.1
        ca = np.random.normal(mu, sigma, size=[points_amount, 1]) * np.expand_dims(w_h[:, 0], axis=1) * np.expand_dims(
            w_h[:, 1], axis=1)

        res = np.hstack((w_h, ca, d, np.ones((points_amount, 1)) * height, np.ones((points_amount, 1)) * angle,
                         np.zeros((points_amount, 1))))

        iter_data = pd.DataFrame(res, columns=['w_rw', 'h_rw', 'c_a_rw', 'd', 'y_rw', 'angle', 'o_class'])
        noise = noise.append(iter_data)
        it += 1
        logging.info(float(it) / lens * 100)

    except ValueError:
        print("No such angle {} or height {}".format(angle, height))


noise.to_csv("new_noises_{}.csv".format(points_amount))