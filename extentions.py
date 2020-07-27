import os
import numpy as np
import cv2
import logging


import conf

logger = logging.getLogger('detect.ext')


class SaveData(object):
    def __init__(self, flag, scaled_calib_mtx, scaled_target_mtx, dist):
        if flag:
            self.fd = open(os.path.join(conf.OUT_DIR, 'h{}_a{}.csv'.format(conf.HEIGHT, conf.ANGLE)), 'w')
            self.fmt = '%d,%d,%.2f,%.2f,%.2f,%.1f,%.2f,%d,%d,%d,%d,%d,%d,%d,%.2f,%d'
            self.fd.write("img,o_num,rw_w,rw_h,rw_ca,rw_z,rw_x,x,y,w,h,ca,p2x,p2y,o_prob,o_class\n")

            self.calib_mtx = scaled_calib_mtx
            self.target_mtx = scaled_target_mtx
            self.dist = dist

            self.color_map = {0: (255, 255, 255), 1: (0, 255, 0), 2: (255, 204, 33), 3: (0, 255, 255)}

        else:
            self.write = self.blank_fn
            self.quit = self.blank_fn

    @staticmethod
    def blank_fn(*args, **kwargs):
        pass

    @staticmethod
    def prepare_array_to_save(data, img_num):
        # Add image number and row indices as first two columns to distinguish objects later
        return np.column_stack((np.full(data.shape[0], img_num), np.arange(data.shape[0]), data))

    def write(self, data, img_num, steps, objects, prob_q, av_bin_result):
        data = self.prepare_array_to_save(data, img_num)
        if data.size > 0:
            np.savetxt(self.fd, data, fmt=self.fmt)

        self.write_images(steps, data, img_num, objects, prob_q, av_bin_result)

    def quit(self):
        self.fd.close()

    def draw_rects(self, img, data_frame, padding):
        for row in data_frame.tolist():
            o_class = row[-1]
            # if o_class == 0:
            #     continue

            o_prob = row[-2]

            o_class_nm = conf.o_class_mapping[o_class]
            x, y, w, h = [int(param) for param in row[7:11]]
            x += padding
            y += padding
            p1 = (x, y)
            p2 = (x + w, y + h)
            color = self.color_map[o_class]
            x_rw, y_rw, z_rw = row[6], conf.HEIGHT, row[5]

            cv2.rectangle(img, p1, p2, color, 1)
            cv2.putText(img, '({0:.2f},{1:.2f},{2:.2f})'.format(x_rw, y_rw, z_rw), (x, p2[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
            cv2.putText(img, '{0:.2f} {1}'.format(o_prob, o_class_nm), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1,
                        cv2.LINE_AA)

    @staticmethod
    def draw_tracking(img, objects, prob_q, padding):
        if len(objects) == 0:
            return

        for obj_key in prob_q.keys():
            average_prob = np.mean(prob_q[obj_key])
            row = objects[obj_key]
            x, y, w, h = [int(param) for param in row[7:11]]
            x += padding
            y += padding

            cv2.putText(img, '{0} '.format(obj_key), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 0, 255), 1, cv2.LINE_AA)
            if len(prob_q[obj_key]) == 3:
                cv2.putText(img, '{0:.2f}'.format(average_prob), (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 0, 255), 1, cv2.LINE_AA)

    @staticmethod
    def add_padding(image, padding_size):
        h_pad = np.zeros((padding_size, conf.RES[0]), np.uint8)
        stack1 = np.vstack((h_pad, image, h_pad))

        v_pad = np.zeros((stack1.shape[0], padding_size), np.uint8)
        stack2 = np.hstack((v_pad, stack1, v_pad))

        return stack2

    @staticmethod
    def draw_lamp_status(img, lamp_status):
        if lamp_status:
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 255), 3)

    def write_images(self, steps, frame, img_num, objects, prob_q, av_bin_result):
        for key, img in steps.items():
            steps[key] = cv2.undistort(img, self.calib_mtx, self.dist, None, self.target_mtx)

        padding = 30
        for key, img in steps.items():
            steps[key] = self.add_padding(img, padding)

        for key, img in steps.items():
            steps[key] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Name frames
        for key, img in steps.items():
            cv2.putText(steps[key], key, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        self.draw_rects(steps['resized_orig'], frame, padding)
        self.draw_tracking(steps['resized_orig'], objects, prob_q, padding)

        h_stack1 = np.hstack((steps['mask'], steps['filtered']))
        h_stack2 = np.hstack((steps['filled'], steps['resized_orig']))

        out_img = np.vstack((h_stack1, h_stack2))

        self.draw_lamp_status(out_img, av_bin_result)

        cv2.imwrite(os.path.join(conf.OUT_DIR, '{}.jpeg'.format(img_num)), out_img)

