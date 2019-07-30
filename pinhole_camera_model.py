import numpy as np
import cv2
import conf


class PinholeCameraModel(object):
    def __init__(self):
        self.obj_dim = [0.7, 1.6, 0.3]
        self.img_res = conf.RESIZE_TO

        self.rw_z_range = np.arange(0, 30, 0.1)
        self.rw_x = 0
        self.rw_y = -(conf.HEIGHT - self.obj_dim[1] / 2.0)
        self.rw_angle = -conf.ANGLE

    def get_cuboid_vertices(self, center_coords):
        # Calculate vertices coordinates of cuboid based on center coordinates
    
        x_l = lambda c, obj: c[0] - obj[0] / 2.0
        x_r = lambda c, obj: c[0] + obj[0] / 2.0
        y_b = lambda c, obj: c[1] - obj[1] / 2.0
        y_t = lambda c, obj: c[1] + obj[1] / 2.0
        z_f = lambda c, obj: c[2] - obj[2] / 2.0
        z_b = lambda c, obj: c[2] + obj[2] / 2.0
    
        cube = [[x_l, y_t, z_f], [x_l, y_b, z_f], [x_r, y_b, z_f], [x_r, y_t, z_f],  # Front rectangle
                [x_l, y_t, z_b], [x_l, y_b, z_b], [x_r, y_b, z_b], [x_r, y_t, z_b]]  # Back rectangle
    
        # Fill the structure by numbers
        cube = np.array([[coord(center_coords, self.obj_dim) for coord in vertex] for vertex in cube])
        # Add one to the each vertex to be in homogeneous coordinates
        cube = np.hstack((cube, np.ones((cube.shape[0], 1))))
    
        return cube

    def get_point_projection(self, w_coords, angle):
        def get_extrinsic_matrix(ang):
            ang = np.radians(ang)

            # R_x(angle)
            r = np.array([[1, 0, 0, 0],
                          [0, np.cos(ang), -np.sin(ang), 0],
                          [0, np.sin(ang), np.cos(ang), 0],
                          [0, 0, 0, 1]])

            return r

        def get_intrinsic_matrix(img_res):
            f_m = 40

            w_img, h_img = img_res
            w_ccd, h_ccd = 36, 26.5

            fx = f_m * w_img / float(w_ccd)
            fy = f_m * h_img / float(h_ccd)

            px = w_img / 2.0
            py = h_img / 2.0

            k = np.array([[fx, 0, px, 0],
                          [0, fy, py, 0],
                          [0, 0, 1, 0]])

            return k
        e = get_extrinsic_matrix(angle)
        k = get_intrinsic_matrix(self.img_res)
    
        cam_coords = np.dot(e, w_coords)
        img_coords_hom = np.dot(k, cam_coords)
    
        # Transform from hom coords
        u, v, _ = img_coords_hom / cam_coords[2]
        # Reverse along y axis
        v = self.img_res[1] - v
    
        return [u, v]

    # Sutherland-Hodgman Polygon-Clipping Algorithm
    def clip_poly(self, polygon):
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
    
        w_img, h_img = self.img_res
    
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
            if not (polygon[0] == polygon[-1]).all():
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

    def get_polygon(self, vertices, angle):
        raw_polygon = np.array([[self.get_point_projection(vertex, angle)] for vertex in vertices], dtype='int32')
        convex_hull = cv2.convexHull(raw_polygon)
        clipped_convex_hull = self.clip_poly(convex_hull)

        return clipped_convex_hull

    @staticmethod
    def calc_geom_params(polygon):
        c_a = cv2.contourArea(polygon)
        b_r = cv2.boundingRect(polygon)
        y = b_r[1] + b_r[3] / 2
        x = b_r[0]
        w = b_r[2]
        h = b_r[3]

        return c_a, x, y, w, h

    def get_2d_params(self, x, y, z, angle):
        # Generate a cuboid object
        cuboid = self.get_cuboid_vertices((x, y, z))
        # Get polygons of the cuboid
        polygon = self.get_polygon(cuboid, angle)

        if polygon is None:
            return None
        else:
            return self.calc_geom_params(polygon)

    @staticmethod
    def drop_nones(x, y):
        # Filter both by y
        nones_i = [i for i, e in enumerate(y) if e is None]
        y_f = [j for i, j in enumerate(y) if i not in nones_i]
        x_f = [j for i, j in enumerate(x) if i not in nones_i]

        return x_f, y_f

    def get_ref_val(self, z_ref):
        ref_2d_params = self.get_2d_params(self.rw_x, self.rw_y, z_ref, self.rw_angle)
        if ref_2d_params is not None:
            return ref_2d_params
        else:
            print 'Raise an exception, no such scenario'

    def init_y_regress(self):
        # Train regression
        z_2d_params = [self.get_2d_params(self.rw_x, self.rw_y, z, self.rw_angle) for z in self.rw_z_range]
        z_range_f, z_2d_params = self.drop_nones(self.rw_z_range, z_2d_params)
        z_2d_params = zip(*z_2d_params)
        # y coord on distance
        y_img_d_poly = np.poly1d(np.polyfit(z_2d_params[2], z_range_f, 8))

        return y_img_d_poly

