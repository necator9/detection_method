from __future__ import division

import itertools
import numpy as np
import sys

from pinhole_camera_model import PinholeCameraModel
from synth_data_func import scale_to_size_all, rotate_y, center_obj
from generate_noises_func import gen_exter_noises, gen_dha, \
    get_cuboid_vertices, find_obj_params2, find_obj_params3, find_combinations, generate_dim
import pandas as pd

oh_dim = {"width": (0.3, 0.64), "height": (1.15, 2), "depth": (0.3, 0.8)}

tp = sys.argv[1]

if tp == 'ex':
    # Generate external noises
    ex_noises = gen_exter_noises(oh_dim, 1, 0)
    noises_whd = itertools.product(ex_noises['width'], ex_noises['height'], ex_noises['depth'])

else:
    tp = 'in'
    # Generate internal noises
    combi = find_combinations(oh_dim, 1, 0)
    combi = generate_dim(combi)
    noises_whd = itertools.chain(*[itertools.product(comb['width'], comb['height'], comb['depth'])
                                   for comb in combi])

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
df_data.to_csv('noises_{}.csv'.format(tp))

























































# combi = find_combinations(oh_dim)
# combi = generate_dim(combi)
#
# gener = [itertools.product(comb['width'], comb['height'], comb['depth']) for comb in combi]
#
# final_gens, len0 = zip(*[gen_dha(gen) for gen in gener])
# # print final_gens, len0
# # # 7140 * 330 * 10 * 4 / 1024 / 1024