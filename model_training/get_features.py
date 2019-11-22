from __future__ import division

import itertools
import numpy as np
import pandas as pd
import sys

from synth_data_func import parse_obj_file1, center_obj, get_kernel_size, find_obj_params5, \
    rotate_y, scale_to_size_all, scale_to_size
from pinhole_camera_model import PinholeCameraModel


obj_info = {'woman_1': {'height': [1.4, 1.95, 10], 'width': [0, 0, 1], 'depth': [0, 0, 1],
                        'file': 'woman-1.obj', 'flipZ': True, 'rotate': [0, 90, 5], 'o_class': 1},
            'thin_man': {'height': [1.4, 2, 10], 'width': [0.5, 0.7, 4], 'depth': [0.8, 0.8, 1],
                         'file': 'running-boy.obj', 'flipZ': True, 'rotate': [0, 0, 1], 'o_class': 1},
            'short_man': {'height': [1.3, 1.6, 6], 'width': [0.7, 0.9, 4], 'depth': [0.8, 0.8, 1],
                          'file': 'running-boy.obj', 'flipZ': True, 'rotate': [0, 0, 1], 'o_class': 1},
            'short_group': {'height': [1.35, 1.57, 4], 'width': [1.05, 1.4, 4], 'depth': [0.8, 0.8, 1],
                            'file': 'pair-1.obj', 'flipZ': True, 'rotate': [0, 0, 1], 'o_class': 2},
            'pair_1': {'height': [1.65, 2.1, 8], 'width': [0, 0, 1], 'depth': [0, 0, 1],
                       'file': 'pair-1.obj', 'flipZ': True, 'rotate': [0, 0, 1], 'o_class': 2},
            'pair_2': {'height': [1.65, 2.1, 8], 'width': [0, 0, 1], 'depth': [0, 0, 1],
                       'file': 'pair-2.obj', 'flipZ': True, 'rotate': [90, 90, 1], 'o_class': 2},
            'tall_group_1': {'height': [1.9, 2, 3], 'width': [0, 0, 1], 'depth': [0, 0, 1],
                             'file': 'pair-1.obj', 'flipZ': True, 'rotate': [90, 90, 1], 'o_class': 2},
            'tall_group_2': {'height': [2.0, 2.05, 3], 'width': [0, 0, 1], 'depth': [0, 0, 1],
                             'file': 'woman-1.obj', 'flipZ': True, 'rotate': [0, 90, 3], 'o_class': 2},
            'cyclist_1': {'height': [1.65, 2.1, 9], 'width': [0, 0, 1], 'depth': [0, 0, 1],
                          'file': 'cyclist-1.obj', 'flipZ': False, 'rotate': [70, 90, 3], 'o_class': 3},
            'disp_car': {'height': [1.35, 2, 15], 'width': [1.8, 1.8, 1], 'depth': [3.8, 5, 15],
                         'file': 'car-3.obj', 'flipZ': True, 'rotate': [90, 90, 1], 'o_class': 4}}

working_obj = obj_info[sys.argv[1]]

vertices, faces = parse_obj_file1(working_obj['file'], working_obj['flipZ'])
obj = np.copy(vertices)

hh_range = np.linspace(*working_obj['height'])
ww_range = np.linspace(*working_obj['width'])
zz_range = np.linspace(*working_obj['depth'])

scale = scale_to_size if sum(ww_range + zz_range) == 0 else scale_to_size_all
dimensions = list(itertools.product(ww_range, hh_range, zz_range))

# cam_angle = np.arange(0, -70, -5)
# cam_angle = [-13, -16, -21]
cam_angle = [-16]


x_range = np.arange(-8, 8, 2)
# y_range = np.arange(-2, -7, -0.2)
# y_range = [-3, -3.1, -4.98]
y_range = [-4.98]

z_range = np.arange(1, 30, 0.3)

rotate_y_angle_range = np.linspace(*working_obj['rotate'])

thr_range = np.linspace(7, 15, 2)
iter_params = list(itertools.product(x_range, y_range, z_range, thr_range))
data = []

lens = np.prod([len(i) for i in (hh_range, ww_range, zz_range, cam_angle, x_range, y_range, z_range, rotate_y_angle_range,
                                 thr_range)])
print ("total iterations: {}".format(lens))
it = 0

img_res = [1280, 720]
f_l = 3.6
w_ccd = 3.4509432207429906
h_ccd = 1.937355215491415

try:
    for angle in cam_angle:
        print(angle)
        pinhole_cam = PinholeCameraModel(rw_angle=angle, f_l=f_l, w_ccd=w_ccd, h_ccd=h_ccd, img_res=img_res)
        for dim in dimensions:
            for rotate_y_angle in rotate_y_angle_range:
                obj = scale(dim, obj)
                obj_size = [obj[:, i].max() - obj[:, i].min() for i in range(3)]
                obj = center_obj(obj)
                kernel_size = get_kernel_size(rotate_y_angle, (1, 13))

                for x, y, z, thr in iter_params:
                    m_o_vert = rotate_y(np.copy(obj), rotate_y_angle, (x, y, z))

                    params = find_obj_params5(m_o_vert, faces, y, pinhole_cam, thr, img_res)
                    if params != None:
                        data.append(params + [x, y, z, rotate_y_angle] + obj_size +
                                    [angle, thr, working_obj['o_class']])

                    it += 1
                    if it % 1000 == 0:
                        print ('{:10.2f}, {}\n'.format(it / lens * 100, it))


except KeyboardInterrupt:
    pass

df_data = pd.DataFrame(data,
                       columns=['d', 'c_a_rw', 'w_rw', 'h_rw', 'extent', 'x', 'y', 'w', 'h', 'c_a_px',
                                'x_rw', 'y_rw', 'z_rw', 'y_rotation', 'width', 'height', 'depth', 'angle',
                                'thr', 'o_class'])

df_data.to_csv('csv_plot/{}.csv'.format(sys.argv[1]))
