# Camera parameters
DEVICE = "/home/ivan/experiments/sources/video/14/selected/test/res/movement_14_4_1.5.mp4"
RESOLUTION = [1280, 720]
FPS = 24

# Virtual camera parameters
VIRTUAL_CAMERA = True

# Detection parameters
IMG_RES = [320, 240]      # Width and height of an image to resize for processing
# IMG_RES = [1280, 720]
DILATE_ITERATIONS = 2    # Dialation is not used


MARGIN = 1

# Saving parameters
SAVE_SINGLE = False
SAVE_VERBOSE = True

WRITE_TO_DB = True
WRITE_TO_PICKLE = False
# TODO Check why the sqlite does not accept "." in path of out database
# OUT_DIR = "/home/ivan/experiments/TZK_january/bicycle_new_dist_method_1"
OUT_DIR = "/home/ivan/experiments/chaos_knn/"
# OUT_DIR = "/home/ivan/experiments/11m_ped_wo/"
# OUT_DIR = "/home/ivan/experiments/random_knn_3.3/"
# OUT_DIR = "/home/ivan/experiments/bicyclist_/"
# OUT_DIR = "/home/ivan/experiments/video_img_color_knn_d_small/"
# OUT_DIR = "/home/ivan/experiments/group_/"


# Timers parameters
TIMERS = True
TIME_WINDOW = 25

# Logging parameters
SHOW_LOGS = True
LOG_LEVEL = "INFO"
# PATH_TO_LOGS = "/root/logs/"
PATH_TO_LOGS = "/home/ivan/logs/"

# # Camera installation parameters
# IN_DIR = '/home/ivan/experiments/sources/TZK_january/3m_4l/random_renamed/'
# ANGLE = 21
# HEIGHT = 3

# IN_DIR = '/home/ivan/experiments/sources/TZK_january/3m_4l/group/'
# ANGLE = 21
# HEIGHT = 3

# IN_DIR = '/home/ivan/experiments/sources/TZK_january/3m_4l/ped/'
# ANGLE = 21
# HEIGHT = 3

# IN_DIR = "/home/ivan/experiments/sources/bicyclist_random_filtered/"
# ANGLE = 13
# HEIGHT = 3.1

# IN_DIR = "/home/ivan/experiments/sources/11m_ped_filtered/"
# ANGLE = 13
# HEIGHT = 3.1

IN_DIR = "/home/ivan/experiments/sources/chaos_renamed/"
ANGLE = 13
HEIGHT = 3.1

# IN_DIR = "/home/ivan/experiments/sources/video/14/selected/test/res/img/"
# ANGLE = 16.4801139558
# HEIGHT = 4.982
