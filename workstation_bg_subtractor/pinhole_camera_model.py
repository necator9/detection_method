import numpy as np
import cv2
import conf


def get_cuboid_vertices(center_coords, obj_dim):
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
    cube = np.array([[coord(center_coords, obj_dim) for coord in vertex] for vertex in cube])
    # Add one to the each vertex to be in homogeneous coordinates
    cube = np.hstack((cube, np.ones((cube.shape[0], 1))))

    return cube


def get_extrinsic_matrix(angle=0):
    angle = np.radians(angle)

    # R_x(angle)
    r = np.array([[1, 0, 0, 0],
                  [0, np.cos(angle), -np.sin(angle), 0],
                  [0, np.sin(angle), np.cos(angle), 0],
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

    K = np.array([[fx, 0,  px, 0],
                  [0,  fy, py, 0],
                  [0,  0,  1,  0]])

    return K


def get_point_projection(w_coords, img_res, angle):
    E = get_extrinsic_matrix(angle)
    K = get_intrinsic_matrix(img_res)

    cam_coords = np.dot(E, w_coords)
    img_coords_hom = np.dot(K, cam_coords)

    # Transform from hom coords
    u, v, _ = img_coords_hom / cam_coords[2]
    # Reverse along y axis
    v = img_res[1] - v

    return [u, v]


# Sutherland-Hodgman Polygon-Clipping Algorithm
def clip_poly(polygon, img_res):
    def IIp(clip_polygon, polygon, i, border=None):
        clip_polygon.append(polygon[i + 1])
#         print "Both inside, p: {}".format(polygon[i + 1])

    def IOi(clip_polygon, polygon, i, border):
        l = line(polygon[i][0], polygon[i + 1][0])
        r = list(intersection(border, l))
        clip_polygon.append([r])
#         print "Inside - Outside, p: {}, p+1 {}, save i: {}".format(polygon[i], polygon[i + 1], r)

    def OO(clip_polygon, polygon, i, border=None):
#         print "Both outside, p: {}, p+1 {}".format(polygon[i], polygon[i + 1])
        pass

    def OIip(clip_polygon, polygon, i, border):
        l = line(polygon[i][0], polygon[i + 1][0])
        r = list(intersection(border, l))
        clip_polygon.append([r])
        clip_polygon.append(polygon[i + 1])
#         print "Outside - Inside, p: {}, p+1 {}, i: {}".format(polygon[i], polygon[i + 1], r)

    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    def intersection(L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return False

    w_img, h_img = img_res

    if (polygon.T[0].min() >= 0 and polygon.T[0].max() <= w_img) and \
            (polygon.T[1].min() >= 0 and polygon.T[1].max() <= h_img):

        return polygon

    cond1, cond2 = lambda a, b: a > b, lambda a, b: a < b
    cond3, cond4 = lambda a, b: a >= b, lambda a, b: a <= b

    routine = [IIp, IOi, OIip, OO]

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


def get_polygon(vertices, img_res, angle):
    raw_polygon = np.array([[get_point_projection(vertex, img_res, angle)] for vertex in vertices], dtype='int32')
    convex_hull = cv2.convexHull(raw_polygon)
    clipped_convex_hull = clip_poly(convex_hull, img_res)

    return clipped_convex_hull


def calc_geom_params(polygon):
    c_a = cv2.contourArea(polygon)
    b_r = cv2.boundingRect(polygon)
    y = b_r[1] + b_r[3] / 2
    x = b_r[0]
    w = b_r[2]
    h = b_r[3]

    return c_a, x, y, w, h


def get_2d_params(x, y, z, angle, img_res):
    obj_dim = [1, 1.7, 0.3]
    # Generate a cuboid object
    cuboid = get_cuboid_vertices((x, y, z), obj_dim)
    # Get polygons of the cuboid
    polygon = get_polygon(cuboid, img_res, angle)

    if polygon is None:
        return None
    else:
        return calc_geom_params(polygon)


def get_ref_val(x_ref, y_ref, z_ref, angle_ref, res_ref):
    ref_2d_params = get_2d_params(x_ref, y_ref, z_ref, angle_ref, res_ref)
    if ref_2d_params is not None:
        return ref_2d_params
    else:
        print 'Raise an exception, no such scenario'

# obj_dim = [1, 1.7, 0.3]
# height = -(conf.HEIGHT - obj_dim[1] / 2.0)
# angle = -conf.ANGLE
# z_range = np.arange(0, 30, 0.1)
# vertexes = [get_cuboid_vertices((0, height, z), obj_dim) for z in z_range]
#
#
# z_range_f, y_list, c_a_list, b_r_list = generate_obj(conf.RESIZE_TO, conf.ANGLE)
# pred_dist_f, c_a_f, w_f, h_f = regress_init(z_range_f, y_list, c_a_list, b_r_list)