from __future__ import division
import numpy as np
import sys

from pinhole_camera_model import PinholeCameraModel
from synth_data_func import scale_to_size_all, rotate_y, center_obj
from generate_noises_func import gen_dha, get_cuboid_vertices, find_obj_params3
import pandas as pd

points_amount = int(sys.argv[1])

w_rg = [0.01, 3]
h_rg = [0.01, 3]
z_rg = [0.01, 3]

# w_rg = [0.01, 5]
# h_rg = [0.01, 5]
# z_rg = [0.01, 5]

# ped = {"width": (0.3, 0.64), "height": (1.15, 2), "depth": (0.3, 0.8)}

# ped = {"width": (0.3, 0.64), "height": (1.3, 2), "depth": (0.3, 0.8)}
# pair = {"width": (0.64, 0.8), "height": (1.3, 2), "depth": (0.3, 1.2)}
cyclist = {"width": (0.3, 0.64), "height": (1.5, 1.9), "depth": (1.4, 1.8)}

# pair = {"width": (0.64, 1.2), "height": (1.15, 2), "depth": (0.3, 0.8)}
# cyclist = {"width": (0.3, 0.64), "height": (1.5, 1.9), "depth": (1.4, 1.8)}


def check_point(candidate, v):
    w_f = candidate['width'][0] < v[0] < candidate['width'][1]
    h_f = candidate['height'][0] < v[1] < candidate['height'][1]
    z_f = candidate['depth'][0] < v[2] < candidate['depth'][1]
    if w_f and h_f and z_f:
        return None
    else:
        return v


noises = []
while True:
    point = np.random.uniform(*w_rg), np.random.uniform(*h_rg), np.random.uniform(*z_rg)
    try:
        # point = check_point(ped, point)
        # point = check_point(pair, point)
        point = check_point(cyclist, point)
    except TypeError:
        continue

    if point is not None:
        noises.append(point)
        if len(noises) > points_amount:
            break

noises_whd = np.array(noises)
ex_noises_xyz, len_xyz = gen_dha(noises_whd)
generator, len0 = gen_dha(noises_whd)


obj = np.copy(get_cuboid_vertices((1, 1, 1)))
c_a_rand_vary = np.arange(0.2, 0.6, 0.01)
data = []

it = 0
try:
    for dim,  x, y, z, angle, rotate_y_angle in generator:
        pinhole_cam = PinholeCameraModel(rw_angle=angle, f_l=40., w_ccd=36., h_ccd=26.5, img_res=[424, 240])

        obj = scale_to_size_all(dim, obj)
        obj_size = [obj[:, i].max() - obj[:, i].min() for i in range(3)]
        obj = center_obj(obj)
        m_o_vert = rotate_y(np.copy(obj), rotate_y_angle, (x, y, z))
        it += 1

        params = find_obj_params3(m_o_vert, y, pinhole_cam, c_a_rand_vary)
        if params is not None:
            data.append(params + [x, y, z, rotate_y_angle] + obj_size + [angle])

        if it % 10000 == 0:
            print it

except KeyboardInterrupt:
    pass

df_data = pd.DataFrame(data,
                       columns=['d', 'c_a_rw', 'w_rw', 'h_rw', 'extent', 'x', 'y', 'w', 'h', 'c_a_px', 'x_rw',
                                'y_rw', 'z_rw', 'y_rotation', 'width', 'height', 'depth', 'angle'])
df_data.to_csv('noises_{}_cycle.csv'.format(points_amount))

























































# combi = find_combinations(oh_dim)
# combi = generate_dim(combi)
#
# gener = [itertools.product(comb['width'], comb['height'], comb['depth']) for comb in combi]
#
# final_gens, len0 = zip(*[gen_dha(gen) for gen in gener])
# # print final_gens, len0
# # # 7140 * 330 * 10 * 4 / 1024 / 1024