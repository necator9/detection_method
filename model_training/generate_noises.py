from __future__ import division
import numpy as np
import sys

from pinhole_camera_model import PinholeCameraModel
from synth_data_func import scale_to_size_all, rotate_y, center_obj
from generate_noises_func import gen_dha, get_cuboid_vertices, find_obj_params3
import pandas as pd

points_amount = int(sys.argv[1])

w_rg = [0.01, 5.5]
h_rg = [0.01, 3]
z_rg = [0.01, 5.5]

ped1 = {"width": (0.4, 0.7), "height": (1.15, 1.2), "depth": (0.4, 0.7)}
ped2 = {"width": (0.35, 0.9), "height": (1.2, 1.4), "depth": (0.35, 0.9)} #
ped5 = {"width": (0.38, 0.9), "height": (1.4, 1.5), "depth": (0.38, 0.9)}
ped3 = {"width": (0.4, 1), "height": (1.5, 1.7), "depth": (0.4, 1)} #
ped = {"width": (0.5, 1.), "height": (1.7, 2.1), "depth": (0.5, 1.)}

pair = {"width": (0.95, 1.6), "height": (1.5, 2.1), "depth": (0.95, 1.6)}

cyclist1 = {"width": (1, 1.7), "height": (1.42, 1.75), "depth": (1, 1.7)}
cyclist2 = {"width": (1, 2), "height": (1.75, 2), "depth": (1, 2)}

car1 = {"width": (3, 4.2), "height": (1.55, 1.8), "depth": (3, 4.2)}
car2 = {"width": (3.2, 4.7), "height": (1.8, 2), "depth": (3.2, 4.7)}
car3 = {"width": (3.5, 4.8), "height": (2, 2.1), "depth": (3.5, 4.8)}


def check_point(candidate, v):
    w_f = not (candidate['width'][0] < v[0] < candidate['width'][1])
    h_f = not (candidate['height'][0] < v[1] < candidate['height'][1])
    z_f = not (candidate['depth'][0] < v[2] < candidate['depth'][1])

    if h_f or (w_f and z_f):
        return v
    else:
        return None


def write_to_csv(header, data):
    df_data = pd.DataFrame(data, columns=['d', 'c_a_rw', 'w_rw', 'h_rw', 'extent', 'x', 'y', 'w',
                                          'h', 'c_a_px', 'x_rw', 'y_rw', 'z_rw', 'y_rotation',
                                          'width', 'height', 'depth', 'angle'])

    with open('noises_{}_all_x0.csv'.format(points_amount), 'a') as f:
        df_data.to_csv(f, header=header)
        header = False

    return header, []


noises = []
while True:
    point = np.random.uniform(*w_rg), np.random.uniform(*h_rg), 0.01#np.random.uniform(*z_rg)
    try:
        point = check_point(ped, point)
        point = check_point(ped1, point)
        point = check_point(ped2, point)
        point = check_point(ped3, point)
        point = check_point(ped5, point)

        point = check_point(pair, point)

        point = check_point(cyclist1, point)
        point = check_point(cyclist2, point)

        point = check_point(car1, point)
        point = check_point(car2, point)
        point = check_point(car3, point)

    except TypeError:
        continue

    if point is not None:
        noises.append(point)
        if len(noises) > points_amount:
            break

noises_whd = np.array(noises)
# generator, len0 = gen_dha(noises_whd, x_range=(0,), y_range=(-6,), z_range=np.arange(1, 30, 1),
#                           angle_range=(-40,), y_rotate_range=(0,))
generator, len0 = gen_dha(noises_whd, x_range=(0,), y_range=np.arange(-2, -7, -0.2),
                          z_range=np.arange(1, 30, 1), angle_range=np.arange(0, -70, -5),
                          y_rotate_range=(0,))


obj = np.copy(get_cuboid_vertices((1, 1, 1)))
c_a_rand_vary = np.arange(0.2, 0.6, 0.01)
data = []
header = True
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
            print(it)

        if it % 1000000 == 0:
            header, data = write_to_csv(header, data)


except KeyboardInterrupt:
    pass


header, data = write_to_csv(header, data)
























































# combi = find_combinations(oh_dim)
# combi = generate_dim(combi)
#
# gener = [itertools.product(comb['width'], comb['height'], comb['depth']) for comb in combi]
#
# final_gens, len0 = zip(*[gen_dha(gen) for gen in gener])
# # print final_gens, len0
# # # 7140 * 330 * 10 * 4 / 1024 / 1024