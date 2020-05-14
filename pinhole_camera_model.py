from __future__ import division
import numpy as np
import logging

logger = logging.getLogger('detect.fe_ext')


class IntrinsicMtx(object):
    def __init__(self, args, vertices, img_points):
        self.img_res, self.f_l, self.sens_dim = args
        self.mtx = np.eye(3, 4)
        np.fill_diagonal(self.mtx, self.f_l * self.img_res / self.sens_dim)
        self.mtx[:, 2] = np.append(self.img_res / 2, 1)  # Append 1 to replace old value in mtx after fill_diagonal

        self.img_points = img_points
        self.vertices = vertices

    def project_to_image(self):
        temp = self.vertices @ self.mtx.T
        self.img_points[:] = np.asarray([temp[:, 0] / temp[:, 2],
                                         temp[:, 1] / temp[:, 2]]).T
        self.img_points[:, 1] = self.img_res[1] - self.img_points[:, 1]  # Reverse along y axis


class RotationMtx(object):
    def __init__(self, key, vertices):
        self.mtx = np.identity(4)
        self.rot_map = {'rx': self.fill_rx_mtx, 'ry': self.fill_ry_mtx, 'rz': self.fill_rz_mtx}
        self.fill_function = self.rot_map[key]
        self.prev_angle = float()

    def build(self, angle):
        #ang = np.deg2rad(angle)
        self.fill_function(np.sin(angle), np.cos(angle))
        return self.mtx

    def fill_rx_mtx(self, a_sin, a_cos):
        self.mtx[1][1] = a_cos
        self.mtx[1][2] = -a_sin
        self.mtx[2][1] = a_sin
        self.mtx[2][2] = a_cos

    def fill_ry_mtx(self, a_sin, a_cos):
        self.mtx[0][0] = a_cos
        self.mtx[0][2] = a_sin
        self.mtx[2][0] = -a_sin
        self.mtx[2][2] = a_cos

    def fill_rz_mtx(self, a_sin, a_cos):
        self.mtx[0][0] = a_cos
        self.mtx[0][1] = -a_sin
        self.mtx[1][0] = a_sin
        self.mtx[1][1] = a_cos


class FeatureExtractor(object):
    """
    Extract object features from given bounding rectangles and contour areas
    :param r_x: # Camera rotation angle about x axis in radians
    :param cam_h: # Ground y coord relative to camera (cam. is origin) in meters
    :param img_res: # Image resolution (width, height) in px
    :param sens_dim: # Camera sensor dimensions (width, height) in mm
    :param f_l: # Focal length in mm
    """
    def __init__(self, r_x, cam_h, img_res, sens_dim, f_l):
        self.r_x = np.deg2rad(r_x, dtype=np.float32)  # Camera rotation angle about x axis in radians
        self.cam_h = np.asarray(cam_h, dtype=np.float32)  # Ground y coord relative to camera (cam. is origin) in meters
        self.img_res = np.asarray(img_res, dtype=np.int16)  # Image resolution (width, height) in px
        self.sens_dim = np.asarray(sens_dim, dtype=np.float32)  # Camera sensor dimensions (width, height) in mm
        self.px_h_mm = self.sens_dim[1] / self.img_res[1]  # Height of a pixel in mm
        self.f_l = np.asarray(f_l, dtype=np.float32)  # Focal length in mm

        # Transformation matrices for 3D reconstruction
        intrinsic_mtx = IntrinsicMtx((self.img_res, self.f_l, self.sens_dim), None, None).mtx
        self.rev_intrinsic_mtx = np.linalg.inv(intrinsic_mtx[:, :-1])  # Last column is not needed in reverse transf.
        rot_x_mtx = RotationMtx('rx', None).build(self.r_x)
        self.rev_rot_x_mtx = np.linalg.inv(rot_x_mtx)

    def extract_features(self, basic_params):
        if basic_params.size == 0:
            return basic_params, basic_params

        b_rect, ca_px = basic_params[:, :4], basic_params[:, -1]
        # * Transform bounding rectangles to required shape
        # Important! Reverse the y coordinates of bound.rect. along y axis before transformations (self.img_res[1] - y)
        px_y_bottom_top = self.img_res[1] - np.stack((b_rect[:, 1] + b_rect[:, 3], b_rect[:, 1]), axis=1)
        # Distances from vertices to img center (horizon) along y axis, in px
        y_bottom_top_to_hor = self.img_res[1] / 2. - px_y_bottom_top
        np.multiply(y_bottom_top_to_hor, self.px_h_mm, out=y_bottom_top_to_hor)  # Convert to mm
        # Find angle between object pixel and central image pixel along y axis
        np.arctan(np.divide(y_bottom_top_to_hor, self.f_l, out=y_bottom_top_to_hor), out=y_bottom_top_to_hor)

        # * Find object distance in real world
        rw_distance = self.estimate_distance(y_bottom_top_to_hor[:, 0])  # Passed arg is angles to bottom vertices
        # * Find object height in real world
        rw_height = self.estimate_height(rw_distance, y_bottom_top_to_hor)

        # * Transform bounding rectangles to required shape
        # Build a single array from left and right rects' coords to compute within single vectorized transformation
        px_x_l_r = np.hstack((b_rect[:, 0], b_rect[:, 0] + b_rect[:, 2]))  # Left and right bottom coords
        # so the [:shape/2] belongs to left and [shape/2:] to the right bound. rect. coordinates
        x_lr_yb_hom = np.stack((px_x_l_r,
                                np.tile(px_y_bottom_top[:, 0], 2),
                                np.ones(2 * px_y_bottom_top.shape[0])), axis=1)

        # * Find object coordinates in real world
        left_bottom, right_bottom = self.estimate_3d_coordinates(x_lr_yb_hom, rw_distance)
        # * Find object width in real world
        rw_width = np.absolute(left_bottom[:, 0] - right_bottom[:, 0])

        # * Find contour area in real world
        rw_rect_a = rw_width * rw_height
        px_rect_a = b_rect[:, 2] * b_rect[:, 3]
        rw_ca = ca_px * rw_rect_a / px_rect_a

        return np.stack((rw_width, rw_height, rw_ca, rw_distance, left_bottom[:, 0] + rw_width / 2), axis=1)

    # Estimate distance to the bottom pixel of a bounding rectangle. Based on assumption that object is aligned with the
    # ground surface. Calculation uses angle between vertex and optical center along vertical axis
    def estimate_distance(self, ang_y_bot_to_hor):
        deg = abs(self.r_x) + ang_y_bot_to_hor
        distance = abs(self.cam_h) / np.where(deg >= 0, np.tan(deg), np.inf)

        return distance

    # Estimate coordinates of vertices in real world
    def estimate_3d_coordinates(self, x_lr_yb_hom, distance):
        # Z cam is a scaling factor which is needed for 3D reconstruction
        z_cam_coords = self.cam_h * np.sin(self.r_x) + distance * np.cos(self.r_x)
        z_cam_coords = np.expand_dims(np.tile(z_cam_coords, 2), axis=0).T
        cam_xlr_yb_h = x_lr_yb_hom * z_cam_coords

        # Transform from image plan to camera coordinate system
        camera_coords = self.rev_intrinsic_mtx @ cam_xlr_yb_h.T
        camera_coords = np.vstack((camera_coords, np.ones((1, camera_coords.shape[1]))))  # To homogeneous form

        # Transform from to camera to real world coordinate system
        rw_coords = self.rev_rot_x_mtx @ camera_coords

        left_bottom, right_bottom = np.split(rw_coords.T, 2, axis=0)  # Split into left/right vertices
        # left_bottom = [[ X,  Y,  Z,  1],  #  - The structure of a returning vertices array, where each row is
        #                [.., .., .., ..]...]    different rectangle. The right_bottom has the same format
        return left_bottom, right_bottom

    # Estimate height of object in real world
    def estimate_height(self, distance, ang_y_bot_top_to_hor):
        angle_between_pixels = np.absolute(ang_y_bot_top_to_hor[:, 0] - ang_y_bot_top_to_hor[:, 1])
        gamma = np.arctan(distance * 1. / abs(self.cam_h))
        beta = np.pi - angle_between_pixels - gamma
        height = np.hypot(abs(self.cam_h), distance) * np.sin(angle_between_pixels) / np.sin(beta)

        return height


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

        br_left_down_2d = np.array([x, y + h, 1])
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