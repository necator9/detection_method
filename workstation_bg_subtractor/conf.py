# Camera parameters
DEVICE = "/dev/video0"
RESOLUTION = (320, 240)
FPS = 7

# Virtual camera parameters
VIRTUAL_CAMERA = True
IN_DIR = "/home/ivan/experiments/sources/5m_ped_filtered"
#IN_DIR = "/home/ivan/experiments/11m_pedestrians/img"
#IN_DIR = "/home/ivan/experiments/16m_pedestrians/img"
#IN_DIR = "/home/ivan/home_ubuntu/test_ir/origin/25.10.17/2"
# Detection parameters
RESIZE_TO = [320, 240]        # Width and height of an image for processing [100, 240]
# F_KERNEL_SIZE = (5, 5)           # Size of elliptical filtering kernel in pixels
F_KERNEL_SIZE = (4, 4)           # Size of elliptical filtering kernel in pixels
DILATE_ITERATIONS = 1
MARGIN = (0, 0)
COEFF_RANGE = (859, 2127)
# COEFF_RANGE = (166, 1662)
# COEFF_RANGE = (10000, 40000)
EXTENT_THRESHOLD = 0.35
BRIGHTNESS_THRESHOLD = 0.2
X_MARGIN = 0

# Saving parameters
SAVE_SINGLE = False
SAVE_VERBOSE = True
WRITE_TO_DB = True
WRITE_TO_PICKLE = False
# OUT_DIR = "/root/test"
OUT_DIR = "/home/ivan/experiments/5m_ped/"
#OUT_DIR = "/home/ivan/experiments/5m_pedestrians_out_clashe_8.0/"
#OUT_DIR = "/home/ivan/experiments/car_bright_0.2/"

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

# Outdated functional
# SHOW_IMG = False
# WRITE_TO_CSV = False




