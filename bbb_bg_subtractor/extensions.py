import logging
import curses
import threading
import time
from numpy import mean
import config
import os
import glob
import argparse

logger = logging.getLogger(__name__)


def parse():
    parser = argparse.ArgumentParser(description='Motion detection setup for Beaglebone Black',
                                     epilog='Hochschule Anhalt, 2017')
    parser.add_argument('--ui', action='store_false',
                        help='Disable command line interface')
    parser.add_argument('--save', action='store_false',
                        help='Disable image saving')
    parser.add_argument('--sync', action='store_false',
                        help='Disable sync with workstation')
    parser.add_argument('--dev', metavar='/path/', default=config.DEVICE,
                        help='Input camera device or file')
    parser.add_argument('--ores', metavar='int', nargs=2, type=int, default=config.ORIG_IMG_RES,
                        help='Resolution of an image to capture from camera (width and height)')
    parser.add_argument('--pres', metavar='int', nargs=2, type=int, default=config.PROC_IMG_RES,
                        help='Resolution of an image for processing (resize, process and save)')
    parser.add_argument('--ofps', metavar='int', type=int, default=config.FPS,
                        help='FPS of a camera')
    parser.add_argument('--fkernel', metavar='int', nargs=2, type=int, default=config.F_KERNEL_SIZE,
                        help='Size of elliptical filtering kernel in pixels')
    parser.add_argument('--osize', metavar='int', type=int, default=config.D_OBJ_SIZE,
                        help='Object size to be detected')
    parser.add_argument('--bsync', metavar='/path/', default=config.BBB_SYNC_DIR,
                        help='Path to synchronizing directory on Beaglebone Black')
    parser.add_argument('--bsave', metavar='/path/', default=config.BBB_IMG_DIR,
                        help='Path to image saving directory on Beaglebone Black')
    parser.add_argument('--wsync', metavar='/path/', default=config.W_SYNC_DIR,
                        help='Path to synchronizing directory on workstation')
    parser.add_argument('--wuip', metavar='username@ip', default=config.W_USER_IP,
                        help='Workstation username@ip')
    parser.add_argument('--wport', metavar='int', default=config.W_PORT,
                        help='Workstation ssh port')
    args = parser.parse_args()

    if not args.ui:
        config.UI = False
    if not args.save:
        config.IMG_SAVE = False
    if not args.sync:
        config.SYNC = False
    config.DEVICE = args.dev
    config.ORIG_IMG_RES = args.ores
    config.PROC_IMG_RES = args.pres
    config.FPS = args.ofps
    config.F_KERNEL_SIZE = args.fkernel
    config.D_OBJ_SIZE = args.osize
    config.BBB_SYNC_DIR = args.bsync
    config.BBB_IMG_DIR = args.bsave
    config.W_SYNC_DIR = args.wsync
    config.W_USER_IP = args.wuip
    config.W_PORT = args.wport


class Display(threading.Thread):
    def __init__(self, st_event):
        super(Display, self).__init__(name="Display")
        self.running = False
        self.stop_event = st_event
        self.screen = curses.initscr()
        self.w_bg = curses.newwin(25, 89, 0, 0)
        self.w_set_params = curses.newwin(11, 55, 2, 1)
        self.w_sync = curses.newwin(10, 55, 13, 1)
        self.w_detect = curses.newwin(3, 30, 7, 57)
        self.w_statistic = curses.newwin(8, 30, 10, 57)

    def run(self):
        logger.info("UI has started")
        self.running = True
        self.window_bg()
        self.window_set_params()
        self.window_statistic()
        self.window_sync()
        i = 0
        while self.running:
            self.window_detect()
            if i > 20:
                self.window_statistic()
                i = 0
            i += 1
            time.sleep(0.1)

    def window_bg(self):
        self.w_bg.border(0)
        self.w_bg.addstr(1, 35, "Moving objects detection", curses.A_BOLD)
        self.w_bg.addstr(23, 1, "To exit press Ctrl^C")
        self.w_bg.refresh()

    def window_set_params(self):
        self.w_set_params.border(1)
        self.w_set_params.addstr(1, 1, "Set parameters", curses.A_BOLD)
        self.w_set_params.addstr(3, 1, "Input device used:")
        self.w_set_params.addstr(3, 32, "%s" % config.DEVICE, curses.A_BOLD)
        self.w_set_params.addstr(4, 1, "Original image resolution:")
        self.w_set_params.addstr(4, 32, "%sx%s" % (config.ORIG_IMG_RES[0], config.ORIG_IMG_RES[1]), curses.A_BOLD)
        self.w_set_params.addstr(5, 1, "Processing image resolution:")
        self.w_set_params.addstr(5, 32, "%sx%s" % (config.PROC_IMG_RES[0], config.PROC_IMG_RES[1]), curses.A_BOLD)
        self.w_set_params.addstr(6, 1, "Capturing frequency, FPS:")
        self.w_set_params.addstr(6, 32, "%s" % config.FPS, curses.A_BOLD)
        self.w_set_params.addstr(7, 1, "Filtering kernel size:")
        self.w_set_params.addstr(7, 32, "%sx%s" % (config.F_KERNEL_SIZE[0], config.F_KERNEL_SIZE[1]), curses.A_BOLD)
        self.w_set_params.addstr(8, 1, "Object size to be detected:")
        self.w_set_params.addstr(8, 32, "%s" % config.D_OBJ_SIZE, curses.A_BOLD)
        self.w_set_params.addstr(9, 1, "Save detected movement:")
        self.w_set_params.addstr(9, 32, "%s" % config.IMG_SAVE, curses.A_BOLD)
        self.w_set_params.refresh()

    def window_sync(self):
        self.w_sync.border(1)
        self.w_sync.addstr(1, 1, "Sync parameters", curses.A_BOLD)
        self.w_sync.addstr(3, 1, "Sync with workstation:")
        self.w_sync.addstr(3, 32, "%s" % config.SYNC, curses.A_BOLD)
        self.w_sync.addstr(4, 1, "BBB sync directory:")
        self.w_sync.addstr(4, 32, "%s" % config.BBB_SYNC_DIR, curses.A_BOLD)
        self.w_sync.addstr(5, 1, "Workstation sync directory:")
        self.w_sync.addstr(5, 32, "%s" % config.W_SYNC_DIR, curses.A_BOLD)
        self.w_sync.addstr(6, 1, "Image saving directory:")
        self.w_sync.addstr(6, 32, "%s" % config.BBB_IMG_DIR, curses.A_BOLD)
        self.w_sync.addstr(7, 1, "Workstation user@ip:")
        self.w_sync.addstr(7, 32, "%s" % config.W_USER_IP, curses.A_BOLD)
        self.w_sync.addstr(8, 1, "Workstation port:")
        self.w_sync.addstr(8, 32, "%s" % config.W_PORT, curses.A_BOLD)
        self.w_sync.refresh()

    def window_detect(self):
        self.w_detect.border(1)
        self.w_detect.addstr(1, 1, "Detection status", curses.A_BOLD)
        self.w_detect.addstr(1, 22, "%s " % config.MOTION_STATUS, curses.A_BOLD)
        self.w_detect.refresh()

    def window_statistic(self):
        mean_gr_t = round(mean(config.T_GRABBER), 3)
        mean_d_t = round(mean(config.T_DETECTOR), 3)
        mean_it_time = mean_gr_t + mean_d_t
        if mean_it_time > 0:
            mean_fps = round(1 / mean_it_time, 3)
        else:
            mean_fps = 0

        self.w_statistic.border(1)
        self.w_statistic.addstr(1, 1, "Statistics", curses.A_BOLD)
        self.w_statistic.addstr(3, 1, "Capturing time, s:")
        self.w_statistic.addstr(3, 22, "%s " % mean_gr_t, curses.A_BOLD)
        self.w_statistic.addstr(4, 1, "Detection time, s:")
        self.w_statistic.addstr(4, 22, "%s " % mean_d_t, curses.A_BOLD)
        self.w_statistic.addstr(5, 1, "Iteration time, s:")
        self.w_statistic.addstr(5, 22, "%s " % mean_it_time, curses.A_BOLD)
        self.w_statistic.addstr(6, 1, "Processing FPS:")
        self.w_statistic.addstr(6, 22, "%s " % mean_fps, curses.A_BOLD)
        self.w_statistic.refresh()

    def quit(self):
        self.running = False
        self.stop_event.clear()
        curses.endwin()
        logger.info("UI has quit")


def check_dir():
    if not os.path.isdir(config.BBB_SYNC_DIR):
        logger.error("No such directory: %s" % config.BBB_SYNC_DIR)
        return False
    if not os.path.isdir(config.BBB_IMG_DIR):
        logger.error("No such directory: %s" % config.BBB_IMG_DIR)
        return False
    else:
        return True


def clear_dir():
    files_n = len(glob.glob(config.BBB_IMG_DIR + "*"))
    if files_n > 0:
        os.system("rm " + config.BBB_IMG_DIR + "*")
        logger.info("Previous files are removed in dir: %s" % config.BBB_IMG_DIR)
    else:
        logger.info("No images detected in dir: %s" % config.BBB_IMG_DIR)


























