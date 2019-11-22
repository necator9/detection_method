from camera_parameters import cameras

# Camera parameters
cam = cameras['camera_1']
ANGLE = cam['angle']
HEIGHT = cam['height']
FL = cam['fl']
WCCD = cam['wccd']
HCCD = cam['hccd']
RESOLUTION = cam['img_res']
FPS = 24

# Device is either /dev/videoX or folder containing images when VIRTUAL_CAMERA == True
DEVICE = '/home/ivan/experiments/sources/clf_test/night/added_to_dataset/sc_2_parking_pg_02/src_320x240_grayscale_corrected/'
OUT_DIR = '/home/ivan/experiments/sources/clf_test/night/added_to_dataset/sc_2_parking_pg_02/res_320x240_wo_ca'
VIRTUAL_CAMERA = True

# Pre-processing parameters
COLOR = cam['color']
CLAHE_LIMIT = 3         # Clahe contrast adjustment for grayscale images only (COLOR = 0)
BGS_METHOD = 'MOG2'
BG_THR = 16
DILATE_ITERATIONS = 1
MARGIN = 1

# Saving parameters
SAVE_SINGLE = False
SAVE_VERBOSE = True

WRITE_TO_DB = True
WRITE_TO_PICKLE = False
# TODO Check why the sqlite does not accept "." in path of out database

# Timers parameters
TIMERS = True
TIME_WINDOW = 25

# Logging parameters
SHOW_LOGS = True
LOG_LEVEL = "INFO"
PATH_TO_LOGS = "/home/ivan/logs/"
