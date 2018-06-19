# Camera parameters
DEVICE = "/dev/video0"
RESOLUTION = (340, 240)
FPS = 7

# Virtual camera parameters
VIRTUAL_CAMERA = True
IN_DIR = "/home/ivan/test_ir/origin/05.10.17/1"

# Detection parameters
RESIZE_TO = [1280, 240]        # Width and height of an image for processing [100, 240]
# F_KERNEL_SIZE = (5, 5)           # Size of elliptical filtering kernel in pixels
F_KERNEL_SIZE = (4, 4)           # Size of elliptical filtering kernel in pixels
DILATE_ITERATIONS = 1
MARGIN = (0, 0)
# COEFF_RANGE = (1200, 12000)
# COEFF_RANGE = (166, 1662)
COEFF_RANGE = (10000, 40000)
EXTENT_THRESHOLD = 0.5
BRIGHTNESS_THRESHOLD = 0.2
X_MARGIN = 0

# Saving parameters
SAVE_SINGLE = True
SAVE_VERBOSE = False
WRITE_TO_DB = True
WRITE_TO_PICKLE = False
# OUT_DIR = "/root/test"
OUT_DIR = "/home/ivan/test_ir/test"

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
PATH_TO_LOGS = "/home/ivan/test_ir/logs/"

# Outdated functional
# SHOW_IMG = False
# WRITE_TO_CSV = False




