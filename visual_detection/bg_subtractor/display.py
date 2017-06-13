import curses
import time
import threading
import config
from numpy import mean

class Display(threading.Thread):
    def __init__(self, stop_ev):
        super(Display, self).__init__(name="Display")
        # Thread state status flag
        self.running = False
        self.stop_event = stop_ev
        self.myscreen = curses.initscr()
        self.win_bg = curses.newwin(16, 90, 0, 0)
        self.win_bg.border(0)
        self.win_stat = curses.newwin(11, 45, 3, 1)
        self.win_d = curses.newwin(3, 40, 3, 47)
        self.win_statistic = curses.newwin(8, 40, 6, 47)


    # Main thread routine
    def run(self):
        self.win_bg.addstr(1, 35, "Moving objects detection", curses.A_BOLD)
        self.win_bg.addstr(14, 1, "To exit press Ctrl^C")
        self.win_bg.refresh()
        self.static_window()
        self.statistic_window()
        i = 0
        while self.stop_event.is_set():
            self.detect_window()
            if i > 20:
                self.statistic_window()
                i = 0
            i += 1
            time.sleep(0.1)

    def static_window(self):
        self.win_stat.border(1)
        self.win_stat.addstr(1, 1, "Set parameters", curses.A_BOLD)
        self.win_stat.addstr(3, 1, "Input device used:")
        self.win_stat.addstr(3, 30, "/dev/video%s" % config.DEV, curses.A_BOLD)
        self.win_stat.addstr(4, 1, "Original image resolution:")
        self.win_stat.addstr(4, 30, "%sx%s" % (config.IMG_WIDTH, config.IMG_HEIGHT), curses.A_BOLD)
        self.win_stat.addstr(5, 1, "Capturing frequency, FPS:")
        self.win_stat.addstr(5, 30, "%s" % config.FPS, curses.A_BOLD)
        self.win_stat.addstr(6, 1, "Filtering kernel size:")
        self.win_stat.addstr(6, 30, "%sx%s" % (config.FILTERED_OBJ_SIZE[0], config.FILTERED_OBJ_SIZE[1]), curses.A_BOLD)
        self.win_stat.addstr(7, 1, "Object size to be detected:")
        self.win_stat.addstr(7, 30, "%s" % config.DETECTED_OBJ_SIZE, curses.A_BOLD)
        self.win_stat.addstr(8, 1, "Save detected movement:")
        self.win_stat.addstr(8, 30, "%s" % config.IMG_SAVE, curses.A_BOLD)
        self.win_stat.addstr(9, 1, "Path to shared folder:")
        self.win_stat.addstr(9, 30, "%s" % config.PATH_TO_SHARE, curses.A_BOLD)
        self.win_stat.refresh()

    def detect_window(self):
        self.win_d.border(1)
        self.win_d.addstr(1, 1, "Detection status", curses.A_BOLD)
        self.win_d.addstr(1, 30, "%s " % config.MOTION_STATUS, curses.A_BOLD)


        self.win_d.refresh()

    def statistic_window(self):
        mean_gr_t = round(mean(config.T_GRABBER), 3)
        mean_d_t = round(mean(config.T_DETECTOR), 3)
        mean_it_time = mean_gr_t + mean_d_t
        if mean_it_time > 0:
            mean_fps = round(1 / mean_it_time, 3)
        else:
            mean_fps = 0

        self.win_statistic.border(1)
        self.win_statistic.addstr(1, 1, "Statistics", curses.A_BOLD)
        self.win_statistic.addstr(3, 1, "Mean capturing time, s:")
        self.win_statistic.addstr(3, 30, "%s " % mean_gr_t, curses.A_BOLD)
        self.win_statistic.addstr(4, 1, "Mean detection time, s:")
        self.win_statistic.addstr(4, 30, "%s " % mean_d_t, curses.A_BOLD)
        self.win_statistic.addstr(5, 1, "Mean iteration time, s:")
        self.win_statistic.addstr(5, 30, "%s " % mean_it_time, curses.A_BOLD)
        self.win_statistic.addstr(6, 1, "Mean FPS:")
        self.win_statistic.addstr(6, 30, "%s " % mean_fps, curses.A_BOLD)
        self.win_statistic.refresh()

    # Stop and quit the thread operation
    def quit(self):
        self.running = False
        self.stop_event.clear()
        curses.endwin()

#def mean_detector():

    # # Timing calculations
    # T_GRABBER = round(mean(T_GRABBER), 3)
    # T_DETECTOR = round(mean(T_DETECTOR),
    # 3)
    # mean_it_time = T_GRABBER + T_DETECTOR
    #
    # logger.info("Mean capturing time %s s", T_GRABBER)
    # logger.info("Mean detection time: %s s", T_DETECTOR)
    # logger.info("Mean iteration time %s s" % mean_it_time)
    # logger.info("Mean FPS %s" % round(1 / mean_it_time, 3))


#self.myscreen.getch()






