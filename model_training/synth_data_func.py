from __future__ import division
import numpy as np
import cv2
import pyblur
import os

from pinhole_camera_model import clip_poly


def find_obj_params1(m_o_vert, faces, height, pinhole_cam, kernel_size):
    projections = np.array([pinhole_cam.get_point_projection(vertex) for vertex in m_o_vert], dtype='int32')
    # polygon = cv2.convexHull(projections)
    # polygon = clip_poly(polygon, pinhole_cam.img_res)

    if len(projections) > 0:
        mask = np.zeros((240, 424), np.uint8)

        for face in faces:
            poly = np.array([projections[i - 1] for i in face])
            mask = cv2.fillPoly(mask, pts=[poly], color=255)

        # Create the blur kernel.
        kernel = np.zeros((max(kernel_size), max(kernel_size)))
        center = int((max(kernel_size) - 1) / 2)
        x_left, x_right = center - int((kernel_size[0] / 2)), center + int((kernel_size[0] / 2))
        y_left, y_right = center - int((kernel_size[1] / 2)), center + int((kernel_size[1] / 2))
        kernel[center, x_left:x_right + 1] = 1
        kernel[y_left:y_right + 1, center] = 1
        kernel /= max(kernel_size)

        # Apply the kernel.
        mask = cv2.filter2D(mask, -1, kernel)

        _, mask = cv2.threshold(mask, 75, 255, cv2.THRESH_BINARY)
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
            c_a_rw = c_a_px * rect_area_rw / rect_area_px

            return [d, c_a_rw, w_rw, h_rw, extent, x, y, w, h, c_a_px]


def find_obj_params4(m_o_vert, faces, height, pinhole_cam, rotate_y_angle, thr):
    projections = np.array([pinhole_cam.get_point_projection(vertex) for vertex in m_o_vert], dtype='int32')
    # polygon = cv2.convexHull(projections)
    # polygon = clip_poly(polygon, pinhole_cam.img_res)

    if len(projections) > 0:
        mask = np.zeros((240, 424), np.uint8)

        for face in faces:
            poly = np.array([projections[i - 1] for i in face])
            mask = cv2.fillPoly(mask, pts=[poly], color=255)

        mask = pyblur.LinearMotionBlur(mask, 3, (90 - rotate_y_angle), 'full')
        mask = np.array(mask, np.uint8)

        _, mask = cv2.threshold(mask, thr, 255, cv2.THRESH_BINARY)
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
            c_a_rw = c_a_px * rect_area_rw / rect_area_px

            return [d, c_a_rw, w_rw, h_rw, extent, x, y, w, h, c_a_px]


def parse_obj_file(path):
    step = 39.3701
    with open(path, "r") as fi:
        vertices = np.array([(ln[3:]).split() for ln in fi if ln.startswith("v")], dtype='float') / step
        vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        vertices[:, 2] = np.negative(vertices[:, 2])
        vertices = center_obj(vertices)

        return vertices


def parse_obj_file1(path, flipZ):
    step = 39.3701
    path = os.path.join('obj', path)
    with open(path, "r") as fi:
        lines = fi.readlines()

    vertices = np.array([parse_string(ln) for ln in lines if ln.startswith("v")], dtype='float') / step
    faces = np.array([parse_string(ln) for ln in lines if ln.startswith("f")], dtype='int')

    vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    if flipZ:
        vertices[:, 2] = np.negative(vertices[:, 2])
    vertices = center_obj(vertices)

    return vertices, faces


def parse_string(string):
    spl = [el.split('//') for el in string.split()]
    res = [el[0] for i, el in enumerate(spl) if i != 0]

    return res


def center_obj(vert):
    centers = [np.median([vert[:, 0].max(), vert[:, 0].min()]), vert[:, 1].min(), vert[:, 2].min()]  # xc, ylowest, zc
    vert[:, 0] = vert[:, 0] - centers[0]
    vert[:, 1] = vert[:, 1] - centers[1]
    vert[:, 2] = vert[:, 2] - centers[2]

    return vert


def move_object(coords, vert):
    for i, val in enumerate(coords):
        vert[:, i] = vert[:, i] + val

    return vert


def scale_object(scale_f, vert):
    for i, val in enumerate(scale_f):
        vert[:, i] *= val

    return vert


def r_y_mtx(r_y_angle, t=[0, 0, 0]):
    ang = np.radians(r_y_angle)
    # R_y(angle)
    r = np.array([[np.cos(ang), 0, np.sin(ang), t[0]],
                  [0, 1, 0, t[1]],
                  [-np.sin(ang), 0, np.cos(ang), t[2]],
                  [0, 0, 0, 1]])

    return r


def rotate_y(vertices, y_angle, t):
    r_y = r_y_mtx(y_angle, t)
    rotated_v = np.array([np.dot(r_y, vert) for vert in vertices])

    return rotated_v


def scale_to_size(scale_size, vert):
    dim = [vert[:, i].max() - vert[:, i].min() for i in range(3)]
    #     scale_f = [scale_el / dim if scale_el / dim != 0 else 1 for scale_el, dim in zip(scale_size, dim)]
    scale_f = max([scale_el / dim for scale_el, dim in zip(scale_size, dim)])
    scale_f = [scale_f for i in range(3)]

    sc_obj = scale_object(scale_f, vert)

    return sc_obj


def scale_to_size_all(scale_size, vert):
    dim = [vert[:, i].max() - vert[:, i].min() for i in range(3)]
    scale_f = [scale_el / dim for scale_el, dim in zip(scale_size, dim)]

    sc_obj = scale_object(scale_f, vert)

    return sc_obj


def form_iter_dimension(dimensions):
    out = []
    for i, dim in enumerate(dimensions):
        zeros = [0, 0, 0]
        for val in dim:
            zeros[i] = val
            out.append([zeros[0], zeros[1], zeros[2]])

    return out


def get_kernel_size(angle, min_max_k):
    x1 = int(round_down_to_odd(0.13333333333333 * angle + min(min_max_k)))
    x2 = int(round_down_to_odd(-0.13333333333333 * angle + max(min_max_k)))

    return x1, x2


def round_down_to_odd(f):
    return abs(np.ceil(f) // 2 * 2 - 1)