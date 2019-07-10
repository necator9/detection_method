import numpy as np


class TrainRegression(object):
    def __init__(self, set_a, set_b, angle_range, h_range):
        self.set_a = set_a
        self.set_b = set_b
        self.angle_range = angle_range
        self.angle_amount = xrange(len(self.angle_range))
        self.h_range = h_range
        self.h_amount = xrange(len(self.h_range))

        self.p_poly = None
        self.p_cols = None
        self.p_amount = None

        self.pp_poly = None
        self.pp_cols = None
        self.pp_amount = None

        self.ppp_poly = None

    def calc_p(self, order=3):
        """
        Find the regresstion polynoms for 2 sets of values for given angle and height.
        p_poly structure: p_poly[h][a][p] where h - height, a - angle, p - calculated polunoms
        """
        self.p_poly = [
            [np.polyfit(self.set_a[height][angle], self.set_b[height][angle], order) for angle in self.angle_amount] for
            height in self.h_amount]
        self.p_amount = xrange(len(self.p_poly[0][0]))
        # p_cols structure: p_cols[h][p][a]
        self.p_cols = [[[poly_row[angle_i][coef_i] for angle_i in self.angle_amount] for coef_i in self.p_amount] for
                       poly_row in self.p_poly]

        return self.p_poly

    def calc_pp(self, order=10, recalc=False):
        """
        Calculate pp_poly describing the dependency the p_poly on the angle range for particular height
        Resulting list structure: pp_poly = [[[10 pp_poly for p_poly[0]],[10 pp_poly for p_poly[1]], [...], [...]], [the same for different height: [...], [...], [...], [...]], ...]
        pp_poly[h][p][0][pp] where h - height index, p - index of first stage polynoms (3th order), p - index of second stage polynoms (10th order)
        """
        if recalc or not self.p_poly:
            self.p_poly = self.calc_p()

        # Transform values of p_poly from rows to coloumns for each height
        #         self.pp_poly = [[[np.polyfit(angle_range, np.log(p_poly_col[coef_i]), order)] for coef_i in self.p_amount] for p_poly_col in self.p_cols]
        #         self.pp_poly = [[[np.polyfit(np.log(angle_range), p_poly_col[coef_i], order)] for coef_i in self.p_amount] for p_poly_col in self.p_cols]
        self.pp_poly = [[[np.polyfit(self.angle_range, p_poly_col[coef_i], order)] for coef_i in self.p_amount] for
                        p_poly_col in self.p_cols]

        self.pp_amount = xrange(len(self.pp_poly[0][0][0]))
        self.pp_cols = [[[pp_poly_row_h[p_i][0][pp_i] for pp_poly_row_h in self.pp_poly] for pp_i in self.pp_amount] for
                        p_i in self.p_amount]

        return self.pp_poly

    def calc_ppp(self, order=9, recalc=False):
        if recalc or not self.pp_poly:
            self.pp_poly = self.calc_pp(recalc=True)

        self.ppp_poly = [
            [np.polyfit(self.h_range, self.pp_cols[p_poly_i][pp_poly_i], order) for pp_poly_i in self.pp_amount] for
            p_poly_i in self.p_amount]

        return self.ppp_poly

    def get_f_from_p(self, h_i, a_i):
        f = np.poly1d(self.p_poly[h_i][a_i])

        return f

    def get_f_from_pp(self, h_i, angle):
        f = np.poly1d([np.poly1d(poly[0])(angle) for poly in self.pp_poly[h_i]])

        return f

    def get_f_from_ppp(self, height, angle):
        ppp_f = [[np.poly1d(self.ppp_poly[p_i][pp_i]) for pp_i in self.pp_amount] for p_i in self.p_amount]
        pp_f_h = [np.poly1d([ppp_f[p_i][pp_i](height) for pp_i in self.pp_amount]) for p_i in self.p_amount]
        p_f_angle = np.poly1d([pp_f_h[p_i](angle) for p_i in self.p_amount])
        f = p_f_angle

        return f