# Created by Ivan Matveev at 01.05.20
# E-mail: ivan.matveev@hs-anhalt.de

# Detection algorithm. All the stages of the detection algorithm are called from this module.

import cv2
import numpy as np
import queue
import timeit
import pickle
from collections import deque
from functools import wraps
import logging

import feature_extractor as fe
from pre_processing import PreprocessImg
import saver
from sl_connect import SlAppConnSensor

logger = logging.getLogger('detect.detect')


class Detection(object):
    def __init__(self, stop_ev, orig_img_q, config):
        self.stop_event = stop_ev
        self.orig_img_q = orig_img_q
        self.config = config

        calib_mtx = np.asarray(config['camera_matrix'])
        calib_res = np.asarray(config['base_res'])
        dist = np.asarray(config['dist_coefs']).reshape(1, -1)

        # Handle case when the target matrix is the same as calibration matrix (target matrix is omit in cam config)
        try:
            target_mtx = np.asarray(config['target_matrix'])
            target_res = np.asarray(config['target_res'])
            if target_mtx is None or target_mtx is None:
                raise KeyError
        except KeyError:
            target_mtx = calib_mtx
            target_res = calib_res

        scaled_calib_mtx = self.scale_intrinsic(config['resolution'], calib_res, calib_mtx)
        scaled_target_mtx = self.scale_intrinsic(config['resolution'], target_res, target_mtx)

        self.frame = Frame(scaled_calib_mtx, scaled_target_mtx, dist, config)
        self.mean_tracker = MeanResultTracker(*config['lamp_on_criteria'])

        self.empty = np.empty([0])

        self.time_measurements = list()
        self.time_window = config['time_window']
        self.sl_app_conn = SlAppConnSensor(config['sl_conn']['detect_port'], [config['sl_conn']['sl_port']])
        self.pre_processing = PreprocessImg(config)

        if config['save_csv']:
            self.save_csv = saver.SaveCSV(config['out_dir'])
        if config['save_img'] or config['stream']['enabled']:
            self.save_img = saver.SaveImg(config, scaled_calib_mtx, scaled_target_mtx, dist)

        if any([config['save_csv'], config['save_img'], config['save_img'], config['stream']['enabled']]):
            self.save_flag = True
        else:
            self.save_flag = True

    @staticmethod
    def scale_intrinsic(new_res, base_res, intrinsic):
        scale_f = np.asarray(base_res) / np.asarray(new_res)
        if scale_f[0] != scale_f[1]:
            logger.warning('WARNING! The scaling is not proportional: {}'.format(scale_f))

        intrinsic[0, :] /= scale_f[0]
        intrinsic[1, :] /= scale_f[1]

        return intrinsic

    @staticmethod
    def prepare_array_to_save(data, img_num, av_bin_result, lamp_status):
        # Add image number and row indices as first two columns to distinguish objects later
        return np.column_stack((np.full(data.shape[0], img_num), np.arange(data.shape[0]), data,
                                np.full(data.shape[0], av_bin_result), np.full(data.shape[0], lamp_status)))

    def run(self):
        logger.info("Detection has started")
        steps = dict()

        iterator = 0
        lamp_status = False
        while not self.stop_event.is_set():
            start_time = timeit.default_timer()

            try:
                orig_img = self.orig_img_q.get(timeout=2)

                lamp_event = self.sl_app_conn.check_lamp_status()
                if lamp_event:
                    lamp_status = not lamp_status
                    logger.debug("Skipping the current frame due to the lamp event")
                    self.orig_img_q.get(timeout=2)  # Blank call to skip current frame
                    logger.debug("Recapturing the frame")
                    orig_img = self.orig_img_q.get(timeout=2)  # Recapture image

            except queue.Empty:
                logger.warning("Timeout reached, no items can be received from orig_img_q")
                continue

            steps['resized_orig'], steps['mask'], steps['filtered'], steps['filled'] = \
                self.pre_processing.apply(orig_img, lamp_event)

            try:
                res_data = self.frame.process(steps['filled'])
                binary_result = np.any(res_data[:, -1] > 0)
            except Frame.FrameIsEmpty:
                res_data = self.empty
                binary_result = False

            av_bin_result = self.mean_tracker.update(binary_result)
            if av_bin_result:
                self.sl_app_conn.switch_on_lamp()

            if self.save_flag:
                packed_data = self.prepare_array_to_save(res_data, iterator, av_bin_result, lamp_status)
                if self.config['save_csv']:
                    self.save_csv.write(packed_data)
                if self.config['save_img'] or self.config['stream']['enabled']:
                    self.save_img.write(steps, packed_data, iterator, lamp_status)

            self.time_measurements.append(timeit.default_timer() - start_time)

            iterator += 1

            if iterator % self.time_window == 0:
                mean_fps = round(1 / (sum(self.time_measurements) / self.time_window), 1)
                logger.info("FPS for last {} samples: mean - {}".format(self.time_window, mean_fps))
                logger.info("Processed images for all time: {} ".format(iterator))
                self.time_measurements = list()

        if self.config['save_csv']:
            self.save_csv.quit()

        if self.config['stream']['enabled']:
            self.save_img.quit()

        logger.info('Detection finished, {} images processed'.format(iterator))


class Frame(object):
    class Decorators(object):
        @classmethod
        def check_input_on_empty_arr(cls, decorated):
            """
            Executes some detection stage (e.g. filtering) if passed array is not empty, otherwise interrupts iteration
            :param decorated: detection function
            :return: mutated array of parameters
            """
            @wraps(decorated)
            def wrapper(*args, **kwargs):
                return decorated(*args, **kwargs) if args[1].size > 0 else Frame.FrameIsEmpty.interrupt_cycle()
            return wrapper

        @classmethod
        def check_on_conf_flag(cls, decorated):
            """
            Executes detection function if corresponding parameter in config is true (>0), otherwise returns original
            array of parameters
            :param decorated: detection function
            :return: original array of parameters or mutated array of parameters
            """
            @wraps(decorated)
            def wrapper(*args, **kwargs):
                return decorated(*args) if kwargs['dec_flag'] else args[1]
            return wrapper

    class FrameIsEmpty(Exception):
        """
        Used to interrupt the processing at any stage when no more objects are remaining in the parameters array (e.g
        due to preliminary filtering)
        """

        def __init__(self):
            Exception.__init__(self, 'No objects in frame are present')

        @staticmethod
        def interrupt_cycle():
            raise Frame.FrameIsEmpty

    def __init__(self, scaled_calib_mtx, scaled_target_mtx, dist, config):
        self.angle = config['angle']
        self.height = config['height']
        self.res = config['resolution']

        self.fe_ext = fe.FeatureExtractor(self.angle, self.height, self.res, intrinsic=scaled_target_mtx)

        self.calib_mtx = scaled_calib_mtx
        self.target_mtx = scaled_target_mtx
        self.dist = dist

        self.img_area_px = self.res[0] * self.res[1]
        self.c_ar_thr = config['cont_area_thr']

        self.margin_offset = config['margin']
        self.left_mar, self.right_mar = self.margin_offset, self.res[0] - self.margin_offset
        self.up_mar, self.bot_mar = self.margin_offset, self.res[1] - self.margin_offset

        self.extent_thr = config['extent_thr']
        self.max_dist_thr = config['max_distance']

        all_classifiers = pickle.load(open(config['clf'], "rb"))
        heights = [key for key in all_classifiers.keys() if type(key) != str]  # Filter the poly key out
        # Find the closest value among available heights
        closest_height = min(heights, key=lambda x: abs(x - self.height))
        # All the available angles for a given height in a form of a list
        angles = list(all_classifiers[closest_height])
        closest_angle = min(angles, key=lambda x: abs(x - self.angle))  # Find the closest value among available angles
        self.clf = all_classifiers[closest_height][closest_angle]
        self.poly = all_classifiers['poly']

    @Decorators.check_input_on_empty_arr
    def find_basic_params(self, mask):
        cnts, _ = cv2.findContours(mask, mode=0, method=1)
        c_areas = [cv2.contourArea(cnt) for cnt in cnts]
        b_rects = [cv2.boundingRect(b_r) for b_r in cnts]

        return np.column_stack((b_rects, c_areas))

    @Decorators.check_input_on_empty_arr
    def calc_second_point(self, temp_param):
        p2_x = temp_param[:, 0] + temp_param[:, 2]
        p2_y = temp_param[:, 1] + temp_param[:, 3]

        return np.column_stack((temp_param, p2_x, p2_y)).astype(np.float32)

    @Decorators.check_input_on_empty_arr
    def undistort(self, basic_params):
        p1p2_col = np.ascontiguousarray(basic_params[:, [0, 1, 5, 6]].reshape((basic_params.shape[0] * 2, 1, 2)))
        cv2.undistortPoints(p1p2_col, self.calib_mtx, self.dist, p1p2_col, P=self.target_mtx)
        p1p2 = p1p2_col.reshape((basic_params.shape[0], 4))
        # p1p2[:, 0] = np.where(p1p2[:, 0] < 0, 0, p1p2[:, 0])
        # p1p2[:, 1] = np.where(p1p2[:, 1] < 0, 0, p1p2[:, 1])
        # p1p2[:, 2] = np.where(p1p2[:, 2] > self.right_mar, self.right_mar, p1p2[:, 2])
        # p1p2[:, 3] = np.where(p1p2[:, 3] > self.bot_mar, self.bot_mar, p1p2[:, 3])

        basic_params[:, :2] = p1p2[:, :2]
        basic_params[:, 2:4] = p1p2[:, 2:4] - p1p2[:, :2]
        basic_params[:, [5, 6]] = p1p2[:, 2:]

    @Decorators.check_on_conf_flag
    @Decorators.check_input_on_empty_arr
    def filter_c_ar(self, basic_params):
        # Filter out small object below threshold
        basic_params = basic_params[basic_params[:, 4] / self.img_area_px > self.c_ar_thr]
        return basic_params

    @Decorators.check_on_conf_flag
    @Decorators.check_input_on_empty_arr
    def filter_extent(self, basic_params):
        basic_params = basic_params[basic_params[:, 4] / (basic_params[:, 2] * basic_params[:, 3]) > self.extent_thr]
        return basic_params

    @Decorators.check_on_conf_flag
    @Decorators.check_input_on_empty_arr
    def filter_margin(self, basic_params):
        margin_filter_mask = ((basic_params[:, 0] > self.left_mar) &  # Built filtering mask
                              (basic_params[:, 5] < self.right_mar) &
                              (basic_params[:, 1] > self.up_mar) &
                              (basic_params[:, 6] < self.bot_mar))

        return basic_params[margin_filter_mask]

    @Decorators.check_on_conf_flag
    @Decorators.check_input_on_empty_arr
    def filter_distance(self, feature_vector):
        # Replace exceeding threshold distances with infinity.
        feature_vector[:, 3] = np.where(feature_vector[:, 3] > self.max_dist_thr, np.inf, feature_vector[:, 3])
        return feature_vector

    @Decorators.check_input_on_empty_arr
    def filter_infinity(self, feature_vector):
        # Filter out infinity distances. Infinities can be already in the feature_vector before filtering by distance!
        feature_vector = feature_vector[np.isfinite(feature_vector[:, 3])]
        return feature_vector

    @Decorators.check_input_on_empty_arr
    def extract_features(self, basic_params):
        return self.fe_ext.extract_features(basic_params)

    @Decorators.check_input_on_empty_arr
    def classify(self, feature_vector):
        poly_features = self.poly.transform(feature_vector)
        o_prob = self.clf.predict_proba(poly_features)
        o_class = np.argmax(o_prob, axis=1)
        o_prob_max = o_prob[np.arange(len(o_class)), o_class]
        return np.column_stack((o_prob_max, o_class))

    def process(self, mask):
        basic_params = self.find_basic_params(mask)
        basic_params = self.calc_second_point(basic_params)
        self.undistort(basic_params)
        # Filtering by object contour area size if filtering by contour area size is enabled
        basic_params = self.filter_c_ar(basic_params, dec_flag=self.c_ar_thr)
        # Filtering by intersection with a frame border if filtering is enabled
        basic_params = self.filter_margin(basic_params, dec_flag=self.margin_offset)
        # basic_params = self.find_contradictory_objects(basic_params, mask)
        basic_params = self.filter_extent(basic_params, dec_flag=self.extent_thr)

        # Get features of the object using its bounding rectangles and contour areas
        feature_vector = np.column_stack((self.extract_features(basic_params), basic_params))
        # Filter by distance to the object if filtering is enabled
        feature_vector = self.filter_distance(feature_vector, dec_flag=self.max_dist_thr)
        feature_vector = self.filter_infinity(feature_vector)
        # Pass informative features only to the classifier
        o_class = self.classify(feature_vector[:, [0, 1, 3]])

        return np.column_stack((feature_vector, o_class))

    @Decorators.check_input_on_empty_arr
    def find_contradictory_objects(self, basic_params, mask):
        print(basic_params)
        splitting_mask = ((basic_params[:, 4] / (basic_params[:, 2] * basic_params[:, 3]) < 0.5) &
                          (basic_params[:, 3] / basic_params[:, 2] < 0.66))

        # Chose problematic indices
        problematic_indices = np.where(splitting_mask)[0]
        if problematic_indices.size > 0:
            for i in problematic_indices:
                if basic_params[i, 4] / self.img_area_px < 0.01:
                    continue
                # Select patch from image
                x1, y1 = basic_params[i, 0], basic_params[i, 1]
                x2, y2 = x1 + basic_params[i, 2], y1 + basic_params[i, 3]
                patch = mask[y1: y2, x1: x2]

                # Split
                patch = self.separate_lighting_spot(patch)
                # Find basic parameters
                b_param = self.find_basic_params(patch)
                b_param = b_param[np.argmax(b_param[:, 4])]
                # Replace problematic object
                b_param[0] += basic_params[i, 0]
                b_param[1] += basic_params[i, 1]
                basic_params[i] = b_param

        return basic_params

    @staticmethod
    def separate_lighting_spot(binary_patch):
        """
        Splitting the object and its light's reflections in specific scenarios of object movement.
        Splitting is performed when the splitting conditions satisfied only:
        width to height ratio > thr1 and extent value > thr2.
        Splitting is designed based on experimental observations, namely:
        1) when the object moves horizontally (relative to a camera frame) the splitting point usually
        lies in a half of a frame (divided along x-axis), where the lighting spot is located.
        2) splitting point is characterized by a significant value of a derivative
        3) the derivative which is corresponding to the splitting point is usually located
        closer to a horizontal frame center
        :param binary_patch: a contradictory segment of a binary image supposed to be split
        :return: split image segment
        """

        nonzero_x = np.count_nonzero(binary_patch, axis=0)  # Find amount of white pixels in columns
        der = np.abs(np.diff(nonzero_x))  # Derivative showing the changes along x-axis
        # Find 2 of the biggest jumps (impulses) of the derivative in left and right sides of a binary image
        middle_point_idx = int(der.shape[0] / 2)
        impulses_idx = np.asarray(
            (np.argmax(der[:middle_point_idx]), np.argmax(der[middle_point_idx:]) + middle_point_idx))
        # Choose one which is closer to center along x-axis.
        distances = np.absolute(impulses_idx - middle_point_idx)
        split_x_index = impulses_idx[np.argmin(distances)]
        # Separate image by drawing vertical line
        binary_patch[:, split_x_index] = 0

        return binary_patch


class MeanResultTracker(object):
    def __init__(self, q_len, true_events):
        self.obj_q = deque(maxlen=q_len)
        self.true_events = true_events

    def update(self, det_result):
        self.obj_q.appendleft(det_result)

        return self.obj_q.count(True) > self.true_events
