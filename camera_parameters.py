import numpy as np

def scale_intrinsic(new_res, base_res, intrinsic):
    scale_f = np.asarray(base_res) / np.asarray(new_res)
    if scale_f[0] != scale_f[1]:
        print('WARNING! The scaling is not proportional', scale_f)

    intrinsic[0, :] /= scale_f[0]
    intrinsic[1, :] /= scale_f[1]

    return intrinsic


def calc_sens_dim(f_l, res, fpx): return f_l * res / fpx


cam_param = {'rpi': {'mtx': np.array([[613, 0., 512],
                                      [0., 613, 354.82125218],
                                      [0., 0., 1.]]),
                     'base_res': (1024, 768),
                     'dist': np.array([[-0.33212234, 0.13364714, 0.0004479, -0.00159172, -0.02811601]])},

             'hd_3000': {'mtx':  np.array([[693.38863768, 0., 339.53274061],
                                          [0., 690.71040995, 236.18033069],
                                          [0., 0., 1.]]),
                         'base_res': (640, 480),
                         'dist': np.array([[0.21584076, -1.58033256, -0.00369491,  0.00366677,  2.94284061]])}}


scene = {'lamp_pole_1': {'angle': -39, 'height': -3.325, 'cam': cam_param['rpi'],
                         'img_res_cap': (1024, 768)},
         'scene_1_TZK': {'angle': -13, 'height': -3.1, 'cam': cam_param['hd_3000'],
                         'img_res_cap': (320, 240)},
         'scene_2_TZK': {'angle': -22, 'height': -3, 'cam': cam_param['hd_3000'],
                         'img_res_cap': (320, 240)}}
