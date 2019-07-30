# Camera parameters
DEVICE = "/dev/video0"
RESOLUTION = (320, 240)
FPS = 7

# Virtual camera parameters
VIRTUAL_CAMERA = True

# Detection parameters
RESIZE_TO = [320, 240]        # Width and height of an image for processing [100, 240]
# F_KERNEL_SIZE = (5, 5)           # Size of elliptical filtering kernel in pixels
F_KERNEL_SIZE = (4, 4)           # Size of elliptical filtering kernel in pixels
DILATE_ITERATIONS = 1  # was 4

MARGIN = (0, 0)
# COEFF_RANGE = (9000, 19000)
COEFF_RANGE = (-0.2, 0.6)
EXTENT_THRESHOLD = 0.35
BRIGHTNESS_THRESHOLD = 0.2
X_MARGIN = 0

# Saving parameters
SAVE_SINGLE = False
SAVE_VERBOSE = True

WRITE_TO_DB = True
WRITE_TO_PICKLE = False
# TODO Check why the sqlite does not accept "." in path of out database
# OUT_DIR = "/home/ivan/experiments/latest_test_random_1/"
# OUT_DIR = "/home/ivan/experiments/diff_16m_ped_filtered/"
OUT_DIR = "/home/ivan/experiments/diff_11m_ped_filtered/"

# Timers parameters
TIMERS = True
TIME_WINDOW = 100

# Logging parameters
SHOW_LOGS = True
LOG_LEVEL = "INFO"
# PATH_TO_LOGS = "/root/logs/"
PATH_TO_LOGS = "/home/ivan/logs/"

# Camera installation parameters
# IN_DIR = "/home/ivan/experiments/sources/TZK_january/3m_4l/random/"
# ANGLE = 21
# HEIGHT = 3


IN_DIR = "/home/ivan/experiments/sources/11m_ped_filtered/"
ANGLE = 13
HEIGHT = 3.1
