import curses
import time
import threading
import config
from numpy import mean

# import pyexiv2


class Display(threading.Thread):
    def __init__(self, stop_ev):
        super(Display, self).__init__(name="Display")
        self.running = False
        self.stop_event = stop_ev
        self.screen = curses.initscr()
        self.win_bg = curses.newwin(16, 90, 0, 0)
        self.win_params = curses.newwin(11, 45, 3, 1)
        self.win_d = curses.newwin(3, 40, 3, 47)
        self.win_statistic = curses.newwin(8, 40, 6, 47)

    def run(self):
        self.win_bg.border(0)
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
        self.win_params.border(1)
        self.win_params.addstr(1, 1, "Set parameters", curses.A_BOLD)
        self.win_params.addstr(3, 1, "Input device used:")
        self.win_params.addstr(3, 30, "/dev/video%s" % config.DEV, curses.A_BOLD)
        self.win_params.addstr(4, 1, "Original image resolution:")
        self.win_params.addstr(4, 30, "%sx%s" % (config.IMG_WIDTH, config.IMG_HEIGHT), curses.A_BOLD)
        self.win_params.addstr(5, 1, "Capturing frequency, FPS:")
        self.win_params.addstr(5, 30, "%s" % config.FPS, curses.A_BOLD)
        self.win_params.addstr(6, 1, "Filtering kernel size:")
        self.win_params.addstr(6, 30, "%sx%s" % (config.FILTERED_OBJ_SIZE[0], config.FILTERED_OBJ_SIZE[1]), curses.A_BOLD)
        self.win_params.addstr(7, 1, "Object size to be detected:")
        self.win_params.addstr(7, 30, "%s" % config.DETECTED_OBJ_SIZE, curses.A_BOLD)
        self.win_params.addstr(8, 1, "Save detected movement:")
        self.win_params.addstr(8, 30, "%s" % config.IMG_SAVE, curses.A_BOLD)
        self.win_params.addstr(9, 1, "Path to shared folder:")
        self.win_params.addstr(9, 30, "%s" % config.PATH_TO_SHARE, curses.A_BOLD)
        self.win_params.refresh()

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

    def quit(self):
        self.running = False
        self.stop_event.clear()
        curses.endwin()


#
#
# class MyImage():
#
#   _filename = None
#   _data = None
#   _metadata = None
#
#   def __init__(self,fname):
#
#     _filename = fname
#     _data = cv.LoadImage(_filename)
#     _metadata = pyexiv2.ImageMetadata(_filename)
#     _metadata.read()
#
#   def addMeta(self,key,value):
#     _metadata[key] = pyexiv2.ExifTag(key, value)
#
#   def delMeta(self,delkey):
#     newdict = {key: value for key, value in some_dict.items()
#                 if value is not delkey}
#
#   def resize(self,newx,newy):
#     tmpimage = cv.CreateMat(newy, newx, cv.CV_8UC3)
#     cv.Resize(_data,tmpimage)
#     _data = tmpimage
#     try: # Set metadata tags to new size if exist!
#       _metadata['Exif.Image.XResolution'] = newx
#       _metadata['Exif.Image.YResolution'] = newy
#     except:
#       pass
#
#   def save(self):
#     cv.SaveImage(_filename,_data)
#     _metadata.write()
#



