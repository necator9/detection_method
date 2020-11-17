import os
import numpy as np
import cv2
import logging
import ffmpeg
import socket

logger = logging.getLogger('detect.ext')


class SaveCSV(object):
    def __init__(self, out_dir):
        self.fd = open(os.path.join(out_dir, 'detected_objects.csv'), 'w')
        self.fmt = '%d,%d,%.2f,%.2f,%.2f,%.1f,%.2f,%d,%d,%d,%d,%d,%d,%d,%.2f,%d,%d,%d'
        self.fd.write("img,o_num,rw_w,rw_h,rw_ca,rw_z,rw_x,x,y,w,h,ca,p2x,p2y,o_prob,o_class,av_bin,lamp\n")

    def write(self, data):
        if data.size > 0:
            np.savetxt(self.fd, data, fmt=self.fmt)

    def quit(self):
        self.fd.close()


class SaveImg(object):
    def __init__(self, config, scaled_calib_mtx, scaled_target_mtx, dist):
        self.config = config

        self.calib_mtx = scaled_calib_mtx
        self.target_mtx = scaled_target_mtx
        self.dist = dist

        self.color_map = {0: (255, 255, 255), 1: (0, 255, 0), 2: (255, 204, 33), 3: (0, 255, 255)}

        width, height = 760, 600
        # width, height = 380, 300

        self.process = (ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height))
                        .output('rtsp://10.33.21.148:8554/{}'.format(socket.gethostname()), vcodec='mpeg4',
                                format='rtsp', video_bitrate='1000', framerate='5')
                        .overwrite_output()
                        .run_async(pipe_stdin=True)
                        )

        self.padding = 30

    # def write(self, data, img_num, steps, objects, prob_q, av_bin_result, lamp_status):
    #
    #     self.write_images(steps, data[:, :-1], img_num, objects, prob_q, av_bin_result, lamp_status)

    def quit(self):
        self.process.stdin.close()
        self.process.wait()

    def write(self, steps, data, iterator):
        print(data)
        for key, img in steps.items():
            steps[key] = cv2.undistort(img, self.calib_mtx, self.dist, None, self.target_mtx)

        for key, img in steps.items():
            steps[key] = self.add_padding(img)

        for key, img in steps.items():
            steps[key] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Name frames
        for key, img in steps.items():
            cv2.putText(steps[key], key, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        self.draw_rects(steps['resized_orig'], data)

        h_stack1 = np.hstack((steps['mask'], steps['filtered']))
        h_stack2 = np.hstack((steps['filled'], steps['resized_orig']))

        out_img = np.vstack((h_stack1, h_stack2))
        cv2.imwrite(os.path.join(self.config['out_dir'], '{}.jpeg'.format(iterator)), out_img)

    def draw_rects(self, img, data):
        data_frame = data[:, [14, 15, 7, 8, 12, 13, 5, 6]]
        data_frame[:, 2:5] += self.padding
        for o_prob, o_class, x, y, p2x, p2y, x_rw, z_rw in data_frame.tolist():
            # if o_class == 0:
            #     continue
            o_class_nm = self.config['o_class_mapping'][o_class]
            color = self.color_map[o_class]
            p1x, p2x, p1y, p2y = int(x), int(p2x), int(y), int(p2y)

            cv2.rectangle(img, (p1x, p1y), (p2x, p2y), color, 1)

            cv2.putText(img, '({0:.2f},{1:.2f},{2:.2f})'.format(x_rw, self.config['height'], z_rw), (p1x, p2y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

            cv2.putText(img, '{0:.2f} {1}'.format(o_prob, o_class_nm), (p1x, p1y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        color, 1, cv2.LINE_AA)

        # for row in data_frame.tolist():
        #     o_class = row[-1]
        #     # if o_class == 0:
        #     #     continue
        #
        #     o_prob = row[-2]
        #
        #     o_class_nm = self.config['o_class_mapping'][o_class]
        #     x, y, w, h = [int(param) for param in row[7:11]]
        #     x += padding
        #     y += padding
        #     p1 = (x, y)
        #     p2 = (x + w, y + h)
        #     color = self.color_map[o_class]
        #     x_rw, y_rw, z_rw = row[6], self.config['height'], row[5]
        #
        #     cv2.rectangle(img, p1, p2, color, 1)
        #     cv2.putText(img, '({0:.2f},{1:.2f},{2:.2f})'.format(x_rw, y_rw, z_rw), (x, p2[1] + 15),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
        #     cv2.putText(img, '{0:.2f} {1}'.format(o_prob, o_class_nm), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
        #                 color, 1, cv2.LINE_AA)

    def add_padding(self, image):
        h_pad = np.zeros((self.padding, self.config['resolution'][0]), np.uint8)
        stack1 = np.vstack((h_pad, image, h_pad))

        v_pad = np.zeros((stack1.shape[0], self.padding), np.uint8)
        stack2 = np.hstack((v_pad, stack1, v_pad))

        return stack2

    @staticmethod
    def draw_av_det_res(img, av_det_status):
        if av_det_status:
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)

    @staticmethod
    def draw_lamp_status(img, lamp_status):
        if lamp_status:
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 255), 3)



    # def write_images(self, steps, frame, img_num, objects, prob_q, av_bin_result, lamp_status):
    #     # for key, img in steps.items():
    #     #     steps[key] = cv2.undistort(img, self.calib_mtx, self.dist, None, self.target_mtx)
    #     #
    #     # for key, img in steps.items():
    #     #     steps[key] = self.add_padding(img, padding)
    #
    #     # for key, img in steps.items():
    #     #     steps[key] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #
    #     # Name frames
    #     # for key, img in steps.items():
    #     #     cv2.putText(steps[key], key, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    #
    #     self.draw_rects(steps['resized_orig'], frame, padding)
    #     # self.process.stdin.write(steps['resized_orig'].astype(np.uint8).tobytes())
    #     h_stack1 = np.hstack((steps['mask'], steps['filtered']))
    #     h_stack2 = np.hstack((steps['filled'], steps['resized_orig']))
    #
    #     out_img = np.vstack((h_stack1, h_stack2))
    #
    #     self.draw_av_det_res(out_img, av_bin_result)
    #     self.draw_lamp_status(out_img, lamp_status)
    #
    #     self.process.stdin.write(out_img.astype(np.uint8).tobytes())

    # self.proc.stdin.write(out_img.tostring())
    # cv2.imwrite(os.path.join(self.config['out_dir'], '{}.jpeg'.format(img_num)), out_img)
