# Camera parameters
DEVICE = "/dev/video0"
RESOLUTION = (320, 240)
FPS = 7

# Virtual camera parameters
VIRTUAL_CAMERA = True
#IN_DIR = "/home/ivan/experiments/sources/angle_vs_obSize/3/3_6/"
IN_DIR = "D:/Ivan/rendering/render_v3/2.7/66"
# IN_DIR = "/home/ivan/experiments/3_6/"
# Detection parameters
RESIZE_TO = [240, 320]        # Width and height of an image for processing [100, 240]
# F_KERNEL_SIZE = (5, 5)           # Size of elliptical filtering kernel in pixels
F_KERNEL_SIZE = (1, 1)           # Size of elliptical filtering kernel in pixels
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
OUT_DIR = "D:/Ivan/test/"

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





