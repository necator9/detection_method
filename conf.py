from camera_parameters import cameras

# Logging parameters
SHOW_LOGS = True
LOG_LEVEL = 10
PATH_TO_LOGS = "/home/ivan/logs/"

# Camera parameters
cam = cameras['camera_1']
ANGLE = cam['angle']
HEIGHT = cam['height']
FL = cam['fl']
WCCD = cam['wccd']
HCCD = cam['hccd']
RES = cam['img_res']
FPS = 24

# Classifier path
CLF_PATH = 'clf_model/detailed_separate_clf_dict.pcl'

# Device is either /dev/videoX or folder containing images when VIRTUAL_CAMERA == True
DEVICE = '/mnt/data_partition/experiments/sources/clf_test/night/added_to_dataset/sc_1_parking_pgc_01/' \
         'src_424x480_grayscale/'
OUT_DIR = '/mnt/data_partition/experiments/sources/clf_test/night/added_to_dataset/sc_1_parking_pgc_01/' \
          'separate_clfs_{}x{}'.format(RES[0], RES[1])

VIRTUAL_CAMERA = True

# Pre-processing parameters
COLOR = cam['color']
CLAHE_LIMIT = 3         # Clahe contrast adjustment for grayscale images only (COLOR = 0)
BGS_METHOD = 'MOG2'
BG_THR = 16
DILATE_ITERATIONS = 1

# Cascade filtering to speed up detection by filtering insignificant objects
# Minimal object cnt area to be considered: object cnt area / RES[0] * RES[1] > CNT_AREA_FILTERING
# Value of zero to disable filtering
CNT_AREA_FILTERING = 0.001  # Chosen 0.0005

# Ignore objects intersecting with frame margin: left img border + MARGIN < obj coordinates < right img border - MARGIN
# Value of zero to disable filtering
MARGIN = 0  # Chosen 1

# Ignore objects which have distance more than MAX_DISTANCE: obj distance > MAX_DISTANCE
# Value of zero to disable filtering
MAX_DISTANCE = 30  # Chosen 30

# Saving parameters
WRITE_IMG = True
WRITE_TO_CSV = True

# Timers parameters
TIMERS = True
TIME_WINDOW = 200


o_class_mapping = {0: 'noise', 1: 'pedestrian', 2: 'group', 3: 'cyclist', 4: 'vehicle'}
