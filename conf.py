# Camera parameters
# DEVICE = "/home/ivan/experiments/sources/clf_test/day/added_to_dataset/pgc_01/movement_14_4_1.5_corrected.mp4"
DEVICE = "/home/ivan/experiments/sources/clf_test/day/added_to_dataset/p_01/corrected_movement_18_46_45.42.mp4"

RESOLUTION = [1280, 720]
FPS = 24

VIRTUAL_CAMERA = False

IMG_RES = [424, 240]      # Width and height of an image to resize for processing
# IMG_RES = [1280, 720]

COLOR = 1
CLAHE_LIMIT = 3         # For grayscale images only (COLOR = 0)
BGS_METHOD = 'MOG2'
DILATE_ITERATIONS = 1

MARGIN = 1

# Saving parameters
SAVE_SINGLE = False
SAVE_VERBOSE = True

WRITE_TO_DB = True
WRITE_TO_PICKLE = False
# TODO Check why the sqlite does not accept "." in path of out database
# OUT_DIR = "/home/ivan/experiments/TZK_january/bicycle_new_dist_method_1"
# OUT_DIR = "/home/ivan/experiments/chaos_knn/"
# OUT_DIR = "/home/ivan/experiments/16m_ped_clache_10_33/"
# OUT_DIR = "/home/ivan/experiments/random_clache_10/"
# OUT_DIR = "/home/ivan/experiments/bicyclist/"
# OUT_DIR = "/home/ivan/experiments/hum_16/"
# OUT_DIR = "/home/ivan/experiments/day_hum_cyclist_01_1280x720/"
# OUT_DIR = "/home/ivan/experiments/day_hum_cyclist_424x240_01/"
# OUT_DIR = "/home/ivan/experiments/day_hum_cyclist_group_424x240_01/"
# OUT_DIR = "/home/ivan/experiments/day_hum_cyclist_424x240_02/"
# OUT_DIR = "/home/ivan/experiments/day_hum_cyclist_424x240_03/"
# OUT_DIR = "/home/ivan/experiments/sources/clf_test/night/parking_pgc_01/res_clache"
# OUT_DIR = "/home/ivan/experiments/sources/clf_test/night/sc_2_parking_pg_01/res_320x240_pcm_mog_clahe3"

#

# OUT_DIR = "/home/ivan/experiments/group_TZK_wo_ca/"
# OUT_DIR = "/home/ivan/experiments/TZK_ped/"


# Timers parameters
TIMERS = True
TIME_WINDOW = 25

# Logging parameters
SHOW_LOGS = True
LOG_LEVEL = "INFO"
# PATH_TO_LOGS = "/root/logs/"
PATH_TO_LOGS = "/home/ivan/logs/"

# Camera installation parameters
# IN_DIR = '/home/ivan/experiments/sources/clf_test/night/sc_2_parking_g_01/src_320x240_grayscale/'
# IN_DIR = '/home/ivan/experiments/sources/clf_test/night/sc_2_parking_c_01/src_320x240_grayscale/'
# OUT_DIR = '/home/ivan/experiments/sources/clf_test/night/sc_2_parking_c_01/res_320x240/'
# ANGLE = 22
# HEIGHT = 3
# FL = 0.73
# WCCD = 0.6
# HCCD = 0.5363504906095236

# IN_DIR = "/home/ivan/experiments/sources/clf_test/night/parking_pgc_01/src_424x480_grayscale/"
# IN_DIR = "/home/ivan/experiments/sources/clf_test/night/parking_c_01/src_424x480_grayscale/"
# ANGLE = 13
# HEIGHT = 3.1
# FL = 40
# WCCD = 36
# HCCD = 26.5


OUT_DIR = '/home/ivan/experiments/sources/clf_test/day/added_to_dataset/p_01/res_424x240_new_clf'
ANGLE = 17 # 17# 16.4801139558 #
HEIGHT = 4.982
FL = 3.6
WCCD = 3.4509432207429906
HCCD = 1.937355215491415
