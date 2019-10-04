# Camera parameters
DEVICE = "/dev/video0"
RESOLUTION = (320, 240)
FPS = 7

# Virtual camera parameters
VIRTUAL_CAMERA = True

# Detection parameters
IMG_RES = [320, 240]      # Width and height of an image to resize for processing
F_KERNEL_SIZE = (3, 3)    # Size of elliptical filtering kernel
DILATE_ITERATIONS = 1     # Dialation is not used


MARGIN = 5

# Saving parameters
SAVE_SINGLE = False
SAVE_VERBOSE = False

WRITE_TO_DB = False
WRITE_TO_PICKLE = False
# TODO Check why the sqlite does not accept "." in path of out database
# OUT_DIR = "/home/ivan/experiments/TZK_january/bicycle_new_dist_method_1"
OUT_DIR = "/home/ivan/experiments/11m_ped_new_dist_method_1/"
# OUT_DIR = "/home/ivan/experiments/random_new_dist_method_1/"
# OUT_DIR = "/home/ivan/experiments/bicyclist_random_filtered/"
# OUT_DIR = "/home/ivan/experiments/16m_ped_new_dist_method_1/"


# Timers parameters
TIMERS = True
TIME_WINDOW = 25

# Logging parameters
SHOW_LOGS = True
LOG_LEVEL = "INFO"
# PATH_TO_LOGS = "/root/logs/"
PATH_TO_LOGS = "/home/ivan/logs/"

# # Camera installation parameters
# IN_DIR = '/home/ivan/experiments/sources/TZK_january/3m_4l/random/'
# ANGLE = 21
# HEIGHT = 3
#
# IN_DIR = "/home/ivan/experiments/sources/bicyclist_random_filtered/"
# ANGLE = 13
# HEIGHT = 3.1

IN_DIR = "/home/ivan/experiments/sources/11m_ped_filtered/"
ANGLE = 13
HEIGHT = 3.1
