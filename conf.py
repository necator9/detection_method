import camera_parameters as cp

# Logging parameters
LOG_LEVEL = 10

# Camera parameters
sc_name = 'scene_1_TZK'
scene = cp.scene[sc_name]
cam = scene['cam']

intrinsic = cp.scale_intrinsic(scene['img_res_cap'], cam['base_res'], cam['mtx'])
dist = cam['dist']

ANGLE = -scene['angle']
HEIGHT = -scene['height']
FL = 2.2
RES = scene['img_res_cap']
WCCD, HCCD = [cp.calc_sens_dim(FL, res_d, fpx_d) for res_d, fpx_d in zip(RES, [intrinsic[0][0], intrinsic[1][1]])]
FPS = 10

# Classifier path
CLF_PATH = 'clf_model/detailed_separate_clf_dict.pcl'

# Device is either /dev/videoX or folder containing images when VIRTUAL_CAMERA == True
DEVICE = '/mnt/data_partition/experiments/sources/clf_test/night/added_to_dataset/sc_1_parking_pgc_01/' \
         'sc_1_parking_pgc_01_4:3_320x240.mp4'

OUT_DIR = '/mnt/data_partition/experiments/sources/clf_test/night/added_to_dataset/sc_1_parking_pgc_01/results/' \
          'temp_1_{}x{}'.format(RES[0], RES[1])

# Pre-processing parameters
COLOR = 0
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
MAX_DISTANCE = 30

# Saving parameters
WRITE_IMG = True
WRITE_TO_CSV = True

# Timers parameters
TIME_WINDOW = 200

o_class_mapping = {0: 'noise', 1: 'pedestrian', 2: 'cyclist', 3: 'vehicle'}
