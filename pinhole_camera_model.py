from __future__ import division
import numpy as np


class PinholeCameraModel(object):
    def __init__(self, rw_angle, img_res, f_l=40., w_ccd=36., h_ccd=26.5):
        self.img_res = img_res
        self.rw_angle = rw_angle

        self.f_l = f_l
        self.w_ccd = w_ccd
        self.h_ccd = h_ccd

        self.e = self.get_extrinsic_matrix()
        self.k = self.get_intrinsic_matrix()

        self.e_inv = np.linalg.inv(self.e)
        self.k_inv = np.linalg.inv(self.k[:, :-1])

    def get_extrinsic_matrix(self):
        ang = np.radians(self.rw_angle)

        # R_x(angle)
        r = np.array([[1, 0, 0, 0],
                      [0, np.cos(ang), -np.sin(ang), 0],
                      [0, np.sin(ang), np.cos(ang), 0],
                      [0, 0, 0, 1]])

        return r

    def get_intrinsic_matrix(self):
        w_img, h_img = self.img_res

        fx = self.f_l * w_img / float(self.w_ccd)
        fy = self.f_l * h_img / float(self.h_ccd)

        px = w_img / 2.0
        py = h_img / 2.0

        k = np.array([[fx, 0, px, 0],
                      [0, fy, py, 0],
                      [0, 0, 1, 0]])

        return k

    def get_point_projection(self, w_coords):
        cam_coords = np.dot(self.e, w_coords)
        img_coords_hom = np.dot(self.k, cam_coords)

        # Transform from hom coords
        u, v, _ = img_coords_hom / cam_coords[2]
        # Reverse along y axis
        v = self.img_res[1] - v

        return [u, v]

    def get_3d_point(self, img_coords, y, z):
        z_cam_coords = (y * np.sin(np.radians(self.rw_angle))) + (z * np.cos(np.radians(self.rw_angle)))

        img_coords_prime = z_cam_coords * img_coords

        camera_coords = self.k_inv.dot(img_coords_prime.T)
        camera_coords[1] = -camera_coords[1]  # Some magic

        camera_coords_prime = np.array([np.append(camera_coords, np.array(1))])
        rw_coords = self.e_inv.dot(camera_coords_prime.T)

        return rw_coords.T[0]

    def get_width(self, y_rw, z_rw, b_rect):
        x, y, w, h = b_rect

        br_left_down_2d = np.array([[x, y + h, 1]])
        br_right_down_2d = np.array([x + w, y + h, 1])
        br_down_2d = [br_left_down_2d, br_right_down_2d]
        br_down_3d = [self.get_3d_point(br_vertex_2d, y_rw, z_rw) for br_vertex_2d in br_down_2d]

        br_left_down_3d = br_down_3d[0][0]
        br_right_down_3d = br_down_3d[1][0]
        br_x_down_list = [br_left_down_3d, br_right_down_3d]

        br_w_3d = abs(max(br_x_down_list) - min(br_x_down_list))

        return br_w_3d

    def find_pixels_angle(self, pix):
        Sh_px = self.img_res[1]
        Sh = self.h_ccd
        FL = self.f_l
        pix = Sh_px - pix
        pxlmm = Sh / Sh_px
        h_px = (Sh_px / 2.) - pix
        h_mm = h_px * pxlmm
        return np.arctan(h_mm / FL)

    def angle_between_pixels(self, pix1, pix2):
        a1 = self.find_pixels_angle(pix1)
        a2 = self.find_pixels_angle(pix2)
        return np.rad2deg(abs(a1 - a2))

    def get_height(self, rw_y, d, b_rect):
        pix1, pix2 = b_rect[1], b_rect[1] + b_rect[3]
        h = abs(rw_y)
        hyp = np.sqrt(h ** 2 + d ** 2)
        gamma = np.rad2deg(np.arctan(d * 1. / h))
        alpha = self.angle_between_pixels(pix1, pix2)
        beta = 180. - alpha - gamma
        return hyp * np.sin(np.deg2rad(alpha)) / np.sin(np.deg2rad(beta))

    def pixels_to_distance(self, rw_y, n):
        h = abs(rw_y)
        r = abs(self.rw_angle)
        Sh_px = float(self.img_res[1])
        Sh = self.h_ccd
        FL = self.f_l

        n = Sh_px - n
        pxlmm = Sh / Sh_px  # print 'pxlmm ', pxlmm
        # h_px = abs((Sh_px / 2.) - n)
        h_px = (Sh_px / 2.) - n
        h_mm = h_px * pxlmm  # print 'hmm ', h_mm
        bo = np.arctan(h_mm / FL)  # print 'bo ', np.rad2deg(bo)
        deg = np.deg2rad(r) + bo
        tan = np.tan(deg) if deg >= 0 else -1.
        # tan = np.tan(deg)
        d = (h / tan)

        return d


# Sutherland-Hodgman Polygon-Clipping Algorithm
def clip_poly(polygon, img_res):
    def ii_p(clip_polygon_, polygon, i, border=None):
        clip_polygon_.append(polygon[i + 1])
        # print "Both inside, p: {}".format(polygon[i + 1])

    def io_i(clip_polygon_, polygon, i, border):
        l = line(polygon[i][0], polygon[i + 1][0])
        r = list(intersection(border, l))
        clip_polygon_.append([r])
        # print "Inside - Outside, p: {}, p+1 {}, save i: {}".format(polygon[i], polygon[i + 1], r)

    def oo(clip_polygon_, polygon, i, border=None):
        # print "Both outside, p: {}, p+1 {}".format(polygon[i], polygon[i + 1])
        pass

    def oi_ip(clip_polygon_, polygon, i, border):
        l = line(polygon[i][0], polygon[i + 1][0])
        r = list(intersection(border, l))
        clip_polygon_.append([r])
        clip_polygon_.append(polygon[i + 1])
        # print "Outside - Inside, p: {}, p+1 {}, i: {}".format(polygon[i], polygon[i + 1], r)

    def line(p1, p2):
        a = (p1[1] - p2[1])
        b = (p2[0] - p1[0])
        c = (p1[0] * p2[1] - p2[0] * p1[1])
        return a, b, -c

    def intersection(l1, l2):
        d = l1[0] * l2[1] - l1[1] * l2[0]
        dx = l1[2] * l2[1] - l1[1] * l2[2]
        dy = l1[0] * l2[2] - l1[2] * l2[0]
        if d != 0:
            x = dx / d
            y = dy / d
            return x, y
        else:
            return False

    w_img, h_img = img_res

    if (polygon.T[0].min() >= 0 and polygon.T[0].max() <= w_img) and \
            (polygon.T[1].min() >= 0 and polygon.T[1].max() <= h_img):
        return polygon

    cond1, cond2 = lambda a, b: a > b, lambda a, b: a < b
    cond3, cond4 = lambda a, b: a >= b, lambda a, b: a <= b

    routine = [ii_p, io_i, oi_ip, oo]

    left_border = line([0, 0], [0, h_img])
    right_border = line([w_img, 0], [w_img, h_img])
    b_border = line([0, h_img], [w_img, h_img])
    t_border = line([0, 0], [w_img, 0])

    l_clip_m = [routine, left_border, 0, 0, [cond3, cond2]]
    r_clip_m = [routine[::-1], right_border, w_img, 0, [cond1, cond4]]
    b_clip_m = [routine, b_border, h_img, 1, [cond4, cond1]]
    t_clip_m = [routine[::-1], t_border, 0, 1, [cond2, cond3]]

    clip_m = [l_clip_m, r_clip_m, b_clip_m, t_clip_m]

    for c_i, clip_case in enumerate(clip_m):
        clip_polygon = list()
        cond1 = clip_case[4][0]
        cond2 = clip_case[4][1]

        cond_m = [[cond1, cond1],
                  [cond1, cond2],
                  [cond2, cond1],
                  [cond2, cond2]]

        # Add last element is equal to first to close the contour
        st = polygon[0] == polygon[-1]
        if not st.all():
            polygon = np.vstack((polygon, [polygon[0]]))

        for i in range(len(polygon) - 1):
            for ind, cond_case in enumerate(cond_m):
                if cond_case[0](polygon[i][0][clip_case[3]], clip_case[2]) and \
                        cond_case[1](polygon[i + 1][0][clip_case[3]], clip_case[2]):
                    clip_case[0][ind](clip_polygon, polygon, i, border=clip_case[1])

        polygon = np.array(clip_polygon, dtype='int32')

        if len(polygon) == 0:
            return None

    return np.array(polygon, dtype='int32')