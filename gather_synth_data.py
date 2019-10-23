from __future__ import division

import itertools
import numpy as np
import pandas as pd
import sys

from synth_data_func import parse_obj_file1, scale_to_size, center_obj, get_kernel_size, find_obj_params1, rotate_y
from pinhole_camera_model import PinholeCameraModel


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


fname = sys.argv[1]
flipZ = str2bool(sys.argv[2])
h_min = float(sys.argv[3])
h_max = float(sys.argv[4])

vertices, faces = parse_obj_file1(fname, flipZ)

obj = np.copy(vertices)

hh_range = np.arange(h_min, h_max, 0.2)
cam_angle = np.arange(0, -70, -5)
x_range = np.arange(-8, 8, 2)
y_range = np.arange(-2, -7, -0.3)
z_range = np.arange(1, 30, 1)
rotate_y_angle_range = np.arange(0, 90, 22)

dimensions = [(0, h, 0) for h in hh_range]


iter_params = list(itertools.product(x_range, y_range, z_range))
data = []

lens = np.prod([len(i) for i in (hh_range, cam_angle, x_range, y_range, z_range, rotate_y_angle_range)])

print "total iterations: {}".format(lens)
it = 0
for angle in cam_angle:
    pinhole_cam = PinholeCameraModel(rw_angle=angle, f_l=40., w_ccd=36., h_ccd=26.5, img_res=[424, 240])
    for dim in dimensions:
        for rotate_y_angle in rotate_y_angle_range:
            obj = scale_to_size(dim, obj)
            obj_size = [obj[:, i].max() - obj[:, i].min() for i in range(3)]
            obj = center_obj(obj)
            kernel_size = get_kernel_size(rotate_y_angle, (1, 13))

            for x, y, z in iter_params:
                m_o_vert = rotate_y(np.copy(obj), rotate_y_angle, (x, y, z))
                it += 1
                params = find_obj_params1(m_o_vert, faces, y, pinhole_cam, kernel_size)
                if params != None:
                    data.append(params + [x, y, z, rotate_y_angle] + obj_size + [angle])

            with open('progress_{}.txt'.format(fname), 'a') as f:
                f.write('{:10.2f}, {}\n'.format(it / lens * 100, it))

df_data = pd.DataFrame(data,
                       columns=['d', 'c_a_rw', 'w_rw', 'h_rw', 'extent', 'x', 'y', 'w', 'h', 'c_a_px', 'x_rw', 'y_rw',
                                'z_rw', 'y_rotation', 'width', 'height', 'depth', 'angle'])
df_data.to_csv('{}.csv'.format(fname))

print len(df_data)