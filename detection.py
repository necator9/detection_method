import threading
import cv2
import numpy as np
import queue

import conf
from extentions import TimeCounter
import logging
import pinhole_camera_model as pcm
from pre_processing import PreprocessImg
import extentions
import tracker

import pickle


logger = logging.getLogger('detect.detect')

all_classifiers = pickle.load(open(conf.CLF_PATH, "rb"))

heights = [key for key in all_classifiers.keys() if type(key) != str]  # Filter the poly key out
closest_height = min(heights, key=lambda x: abs(x - (-conf.HEIGHT)))  # Find the closest value among available heights
angles = list(all_classifiers[closest_height])  # All the available angles for a given height in a form of a list
closest_angle = min(angles, key=lambda x: abs(x - (-conf.ANGLE)))  # Find the closest value among available angles
CLASSIFIER = all_classifiers[closest_height][closest_angle]

poly = all_classifiers['poly']

PINHOLE_CAM = pcm.PinholeCameraModel(rw_angle=-conf.ANGLE, f_l=conf.FL, w_ccd=conf.WCCD, h_ccd=conf.HCCD,
                                     img_res=conf.RES)


class Detection(threading.Thread):
    def __init__(self, stop_ev, data_frame_q, orig_img_q):
        super(Detection, self).__init__(name="detection")
        self.stop_event = stop_ev
        self.img_name = str()

        self.data_frame_q = data_frame_q
        self.orig_img_q = orig_img_q
        self.fe = pcm.FeatureExtractor(-conf.ANGLE, -conf.HEIGHT, conf.RES, (conf.WCCD, conf.HCCD), conf.FL)

        self.timer = TimeCounter("detection_timer")

        self.frame = Frame(self.fe)

        self.empty = np.empty([0])
        self.tracker = tracker.CentroidTracker()

    def run(self):
        logger.info("Detection has started")
        preprocessing = PreprocessImg()
        steps = dict()

        while self.stop_event.is_set():
            self.timer.note_time()

            try:
                orig_img, img_name = self.orig_img_q.get(timeout=2)
            except queue.Empty:
                logger.warning("Timeout reached, no items can be received from orig_img_q")
                continue

            steps['resized_orig'], steps['mask'], steps['filtered'], steps['filled'] = preprocessing.apply(orig_img)

            try:
                res_data = self.frame.process(steps['filled'])
                data_to_save = self.prepare_array_to_save(res_data, int(img_name[: -5]))
                self.data_frame_q.put(data_to_save, block=True)
                #coordinates = np.column_stack((data_to_save[:, 6], data_to_save[:, 5], data_to_save[:, -2]))
                coordinates = data_to_save[data_to_save[:, -1] > 0]

            except FrameIsEmpty:
                data_to_save = self.empty
                coordinates = self.empty

            objects, prob_q = self.tracker.update(coordinates)

            if conf.WRITE_IMG:
                extentions.write_steps(steps, data_to_save, img_name, objects, prob_q)

            self.timer.get_time()

    @ staticmethod
    def prepare_array_to_save(data, img_num):
        # Add image number and row indices as first two columns to distinguish objects later
        return np.column_stack((np.full(data.shape[0], img_num), np.arange(data.shape[0]), data))


class FrameIsEmpty(Exception):
    def __init__(self):
        Exception.__init__(self, 'No object in frame are present')


class Frame(object):
    def __init__(self, fe_ext):
        self.fe_ext = fe_ext

        self.img_area_px = conf.RES[0] * conf.RES[1]
        self.c_ar_thr = conf.CNT_AREA_FILTERING

        self.margin_offset = conf.MARGIN
        self.left_mar, self.right_mar = self.margin_offset, conf.RES[0] - self.margin_offset
        self.up_mar, self.bot_mar = self.margin_offset, conf.RES[1] - self.margin_offset

        self.max_dist_thr = conf.MAX_DISTANCE

        self.first_in = np.ones([1])
        self.empty = np.empty([1])

        self.poly = poly
        self.clf = CLASSIFIER

    def check_on_conf_flag(fun_to_call):
        def wrapper(self, fun_arg, flag):
            return fun_to_call(self, fun_arg) if flag else fun_arg
        return wrapper

    def check_input_on_empty_arr(fun_to_call):
        def wrapper(self, parameters):
            if parameters.size > 0:
                return fun_to_call(self, parameters)
            else:
                raise FrameIsEmpty

        return wrapper

    @check_input_on_empty_arr
    def find_basic_params(self, mask):
        cnts, _ = cv2.findContours(mask, mode=0, method=1)
        c_areas = [cv2.contourArea(cnt) for cnt in cnts]
        b_rects = [cv2.boundingRect(b_r) for b_r in cnts]

        return np.column_stack((b_rects, c_areas)).astype('int32')

    @check_on_conf_flag
    @check_input_on_empty_arr
    def filter_c_ar(self, basic_params):
        # Filter out small object below threshold
        basic_params = basic_params[basic_params[:, -1] / self.img_area_px > self.c_ar_thr]
        return basic_params

    @check_on_conf_flag
    @check_input_on_empty_arr
    def filter_margin(self, basic_params):
        margin_filter_mask = ((basic_params[:, 0] > self.left_mar) &  # Built filtering mask
                              (basic_params[:, 0] + basic_params[:, 2] < self.right_mar) &
                              (basic_params[:, 1] > self.up_mar) &
                              (basic_params[:, 1] + basic_params[:, 3] < self.bot_mar))

        return basic_params[margin_filter_mask]

    @check_on_conf_flag
    @check_input_on_empty_arr
    def filter_distance(self, feature_vector):
        # Replace exceeding threshold distances with infinity.
        feature_vector[:, 3] = np.where(feature_vector[:, 3] > self.max_dist_thr, np.inf, feature_vector[:, 3])
        return feature_vector

    @check_input_on_empty_arr
    def filter_infinity(self, feature_vector):
        # Filter out infinity distances. Infinities can be already in the feature_vector before filtering by distance!
        feature_vector = feature_vector[np.isfinite(feature_vector[:, 3])]
        return feature_vector

    @check_input_on_empty_arr
    def extract_features(self, basic_params):
        return self.fe_ext.extract_features(basic_params)

    @check_input_on_empty_arr
    def classify(self, feature_vector):
        poly_features = self.poly.transform(feature_vector)
        o_prob = self.clf.predict_proba(poly_features)
        o_class = np.argmax(o_prob, axis=1)
        o_prob_max = o_prob[np.arange(len(o_class)), o_class]
        return np.column_stack((o_prob_max, o_class))

    def process(self, mask):
        basic_params = self.find_basic_params(mask)
        # Filtering by object contour area size if filtering by contour area size is enabled
        basic_params = self.filter_c_ar(basic_params, self.c_ar_thr > 0)
        # Filtering by intersection with a frame border if filtering is enabled
        basic_params = self.filter_margin(basic_params, self.margin_offset > 0)
        # Get features of the object using its bounding rectangles and contour areas
        feature_vector = np.column_stack((self.extract_features(basic_params), basic_params))
        # Filter by distance to the object if filtering is enabled
        feature_vector = self.filter_distance(feature_vector, self.max_dist_thr > 0)
        feature_vector = self.filter_infinity(feature_vector)
        # Pass only informative features to classifier
        o_class = self.classify(feature_vector[:, :4])

        return np.column_stack((feature_vector, o_class))


    # def split_object(self, obj_to_split, filled):
    #     def make_split(bin_mask, fill=0.68, tail=0.25): # fill - amount of zeros in coloumn in percent ratio
    #         def calc_split_point(vector):
    #             last_zero_ind, percent = 0, 0.0
    #             zero_indexes, = np.where(vector == 0)
    #
    #             if zero_indexes.size > 0:
    #                 last_zero_ind = zero_indexes.max()
    #                 percent = last_zero_ind / vector.size  # Get relative size of empty area by x axis
    #
    #             return last_zero_ind, percent
    #
    #         rows, coloumns = bin_mask.shape
    #
    #         if coloumns >= rows:
    #             x_mask = np.asarray([0 if (np.count_nonzero(i == 0) / i.size) >= fill else 1 for i in bin_mask.T],
    #                                 dtype='int8')
    #             x_mask_l, x_mask_r = x_mask[:x_mask.size // 2], x_mask[x_mask.size // 2:]
    #
    #             front_ind, front_percent = calc_split_point(x_mask_l)
    #             opposite_ind, opposite_percent = calc_split_point(x_mask_r[::-1])
    #
    #             split_data = [[front_ind, front_percent], [bin_mask.shape[1] - opposite_ind, opposite_percent]]
    #
    #             split_data = zip(*split_data)
    #             max_sp_ind = split_data[1].index(max(split_data[1]))
    #
    #             if split_data[1][max_sp_ind] > tail:
    #                 ind = split_data[0][max_sp_ind]
    #                 bin_mask[:, ind] = 0
    #
    #         return bin_mask
    #
    #     split_img = np.zeros((conf.RES[1], conf.RES[0]), np.uint8)
    #
    #     for obj in obj_to_split:
    #         x, y, w, h = obj.base_rect_ao
    #         split_mask = make_split(filled[y:y + h, x:x + w])
    #         split_img[y:y+h, x:x + w] = split_mask[:, :]
    #
    #     return split_img
