import logging
import curses
import threading
import time
from numpy import mean
import config
import os
import glob

logger = logging.getLogger(__name__)


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


























