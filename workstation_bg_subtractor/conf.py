# Camera parameters
DEVICE = "/dev/video0"
RESOLUTION = (320, 240)
FPS = 7

# Virtual camera parameters
VIRTUAL_CAMERA = True
IN_DIR = "/home/ivan/experiments/sources/TZK_january/2m_2l/magn"
# Detection parameters
RESIZE_TO = [320, 240]        # Width and height of an image for processing [100, 240]
# F_KERNEL_SIZE = (5, 5)           # Size of elliptical filtering kernel in pixels
F_KERNEL_SIZE = (4, 4)           # Size of elliptical filtering kernel in pixels
DILATE_ITERATIONS = 1

MARGIN = (0, 0)
COEFF_RANGE = (1000, 2670)
EXTENT_THRESHOLD = 0.35
BRIGHTNESS_THRESHOLD = 0.2
X_MARGIN = 0

# Saving parameters
SAVE_SINGLE = False
SAVE_VERBOSE = True
WRITE_TO_DB = True
WRITE_TO_PICKLE = False
# TODO Check why the sqlite does not accept "." in path of out database
OUT_DIR = "/home/ivan/experiments/angle_vs_objSize/1_8/0_5"

# Streaming parameters
STREAMING = False
SERVER_TCP_IP = "192.168.4.8"
SERVER_TCP_PORT = 5001

# Timers parameters
TIMERS = True
TIME_WINDOW = 100

# Logging parameters
SHOW_LOGS = True
LOG_LEVEL = "INFO"
# PATH_TO_LOGS = "/root/logs/"
PATH_TO_LOGS = "/home/ivan/logs/"





