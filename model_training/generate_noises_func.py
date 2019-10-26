from __future__ import division

import itertools
import numpy as np
import cv2
import random
from pinhole_camera_model import clip_poly


def find_combinations(o_dim, inf, margin):
    dim_comb = itertools.combinations(o_dim, 2)

    list_dict = []
    for dim in dim_comb:
        named_combinations = {'width': None, 'height': None, 'depth': None}
        named_combinations[dim[0]] = o_dim[dim[0]]
        named_combinations[dim[1]] = o_dim[dim[1]]

        list_dict.append(named_combinations)

    for dict_item in list_dict:
        for key, value in dict_item.iteritems():
            if value is None:
                dict_item[key] = find_noise_ranges(o_dim[key], inf, margin)
    return list_dict


def find_noise_ranges(o_interval, infinity, margin):
    left_inf = min(o_interval) - infinity
    left = min(o_interval) - margin
    right_inf = max(o_interval) + infinity
    right = max(o_interval) + margin

    return (left_inf, left), (right, right_inf)


def generate_dim(combi):
    for comb in combi:
        generate_single_dim(comb)

    return combi


def generate_single_dim(comb):
    for key, value in comb.iteritems():
        try:
            comb[key] = gen_single_range(value)
        except TypeError:
            comb[key] = np.concatenate([gen_single_range(tpl) for tpl in value])

    return comb


def check_val(val):
    return 0.01 if val <= 0 else val


def gen_single_range(rg):
    min_c, max_c = check_val(min(rg)), check_val(max(rg))
    diff = max_c - min_c
    if diff <= 0.5:
        num = 3
    elif 0.5 < diff <= 1:
        num = 5
    else:
        num = 7

    return np.unique(np.linspace(min_c, max_c, num=num))


def gen_dha(whz_gen, x_range=(0,), y_range=np.arange(-2, -7, -0.3), z_range=np.arange(1, 30, 2),
            angle_range=np.arange(0, -70, -5), y_rotate_range=np.arange(0, 90, 45)):
    gen_dha = itertools.product(whz_gen, x_range, y_range, z_range, angle_range, y_rotate_range)
    dha_lengths = len(angle_range) * len(x_range) * len(y_range) * len(z_range) * len(y_rotate_range)

    return gen_dha, dha_lengths


def gen_exter_noises(oh_dim, inf, margin):
    temp_dict = {'width': None, 'height': None, 'depth': None}
    for key, value in oh_dim.iteritems():
        temp_dict[key] = find_noise_ranges(oh_dim[key], inf, margin)

    generate_single_dim(temp_dict)

    return temp_dict


def get_cuboid_vertices(obj_dim):
    # Calculate vertices coordinates of cuboid based on center coordinates

    x_l = lambda c, obj: 0
    x_r = lambda c, obj: obj[0]

    y_b = lambda c, obj: 0
    y_t = lambda c, obj: obj[1]

    z_f = lambda c, obj: 0
    z_b = lambda c, obj: obj[2]

    cube = [[x_l, y_t, z_f], [x_l, y_b, z_f], [x_r, y_b, z_f], [x_r, y_t, z_f],  # Front rectangle
            [x_l, y_t, z_b], [x_l, y_b, z_b], [x_r, y_b, z_b], [x_r, y_t, z_b]]  # Back rectangle

    # Fill the structure by numbers
    cube = np.array([[coord(None, obj_dim) for coord in vertex] for vertex in cube])
    # Add one to the each vertex to be in homogeneous coordinates
    cube = np.hstack((cube, np.ones((cube.shape[0], 1))))

    return cube


def find_obj_params2(m_o_vert, height, pinhole_cam, c_a_rand_vary):
    projections = np.array([pinhole_cam.get_point_projection(vertex) for vertex in m_o_vert], dtype='int32')
    polygon = cv2.convexHull(projections)
    polygon = clip_poly(polygon, pinhole_cam.img_res)

    if polygon is not None:
        #         mask = np.zeros((240, 424), np.uint8)
        #         mask = cv2.fillPoly(mask, pts=[polygon], color=255)
        #         plt.imshow(mask, cmap='gray')
        #         plt.show()

        c_a_px = cv2.contourArea(polygon)

        b_r = x, y, w, h = cv2.boundingRect(polygon)
        d = pinhole_cam.pixels_to_distance(height, y + h)

        h_rw = pinhole_cam.get_height(height, d, b_r)
        w_rw = pinhole_cam.get_width(height, d, b_r)

        rect_area_rw = w_rw * h_rw
        rect_area_px = w * h
        extent = float(c_a_px) / rect_area_px
        c_a_rw = (c_a_px * rect_area_rw / rect_area_px) * random.choice(c_a_rand_vary)

        return [d, c_a_rw, w_rw, h_rw, extent, x, y, w, h, c_a_px]


def find_obj_params3(m_o_vert, height, pinhole_cam, c_a_rand_vary):
    projections = np.array([pinhole_cam.get_point_projection(vertex) for vertex in m_o_vert], dtype='int32')
    polygon = cv2.convexHull(projections)
    # polygon = clip_poly(polygon, pinhole_cam.img_res)

    if len(projections) > 0:
        mask = np.zeros((240, 424), np.uint8)
        mask = cv2.fillPoly(mask, pts=[polygon], color=255)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_a_pxs = [cv2.contourArea(cnt) for cnt in cnts]

        if any(c_a_pxs):
            c_a_px_i = c_a_pxs.index(max(c_a_pxs))
            c_a_px = c_a_pxs[c_a_px_i]

            b_r = x, y, w, h = cv2.boundingRect(cnts[c_a_px_i])
            d = pinhole_cam.pixels_to_distance(height, y + h)

            h_rw = pinhole_cam.get_height(height, d, b_r)
            w_rw = pinhole_cam.get_width(height, d, b_r)

            rect_area_rw = w_rw * h_rw
            rect_area_px = w * h
            extent = float(c_a_px) / rect_area_px
            c_a_rw = ((c_a_px * rect_area_rw) / rect_area_px) * random.choice(c_a_rand_vary)

            return [d, c_a_rw, w_rw, h_rw, extent, x, y, w, h, c_a_px]