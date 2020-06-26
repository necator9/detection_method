# Created by Ivan Matveev at 25.06.20
# E-mail: ivan.matveev@hs-anhalt.de
import numpy as np


def calc_sens_dim(f_l, res, fpx): return f_l * res / fpx


f_mm = float()
img_res = (int(), int())
f_px = (float(), float())

sensor_dim = [calc_sens_dim(f_mm, res_d, fpx_d) for res_d, fpx_d in zip(img_res, f_px)]


