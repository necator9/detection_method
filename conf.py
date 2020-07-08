import camera_parameters as cp

# Logging parameters
LOG_LEVEL = 10

# Camera parameters
# sc_name = 'scene_1_TZK'
sc_name = 'lamp_pole_1'
scene = cp.scene[sc_name]
cam = scene['cam']

intrinsic_target = cp.scale_intrinsic(scene['img_res_cap'], cam['base_res'], cam['mtx'])
intrinsic_orig = cp.scale_intrinsic(scene['img_res_cap'], cam['base_res'], cam['mtx_orig'])
dist = cam['dist']

ANGLE = -scene['angle']
HEIGHT = -scene['height']
FL = 2.2
RES = scene['img_res_cap']
WCCD, HCCD = [cp.calc_sens_dim(FL, res_d, fpx_d) for res_d, fpx_d in zip(RES, [intrinsic_target[0][0], intrinsic_target[1][1]])]
FPS = 10
cxcy = (intrinsic_target[0][2], intrinsic_target[1][2])

# Classifier path
# CLF_PATH = 'clf_model/detailed_separate_clf_dict.pcl'
CLF_PATH = 'clf_model/lamp_pole_1.pcl'

# DEVICE = '/mnt/data_partition/experiments/sources/lighting_pole_1/vid_3_1_4fps_night.mp4'
# DEVICE = '/mnt/data_partition/experiments/sources/lighting_pole_1/vid_1_1_4fps_we.mp4'
# DEVICE = '/mnt/data_partition/experiments/sources/clf_test/night/added_to_dataset/sc_1_parking_pgc_01/sc_1_parking_pgc_01_4:3_320x240.mp4'
# DEVICE = '/mnt/data_partition/experiments/sources/lighting_pole_1/vid_3_cars_selected/car_night_merged.mp4'
DEVICE = '/mnt/data_partition/experiments/sources/lighting_pole_1/tests/car_night_merged_reen2.mkv'
# DEVICE = '/mnt/data_partition/experiments/sources/lighting_pole_1/tests/car_night_merged_reen1.mp4'
# OUT_DIR = '/mnt/data_partition/experiments/sources/lighting_pole_1/results/car_night_merged_1_{}x{}'.format(RES[0], RES[1])
OUT_DIR = '/mnt/data_partition/experiments/test_3{}x{}'.format(RES[0], RES[1])

# Pre-processing parameters
COLOR = 0
CLAHE_LIMIT = 3         # Clahe contrast adjustment for grayscale images only (COLOR = 0)

# Background subtraction parameters
BGS_METHOD = 'MOG2'  # KNN is also available
BG_THR = 16  # 16  # For MOG2 only
DILATE_ITERATIONS = 1
HISTORY = 1500
SHADOWS = True

# Cascade filtering to speed up detection by filtering insignificant objects
# Minimal object cnt area to be considered: object cnt area / RES[0] * RES[1] > CNT_AREA_FILTERING
# Value of zero to disable filtering
CNT_AREA_FILTERING = 0.001  # Chosen 0.0005

EXTENT_THR = 0.2

# Ignore objects intersecting with frame margin: left img border + MARGIN < obj coordinates < right img border - MARGIN
# Value of zero to disable filtering
MARGIN = 0  # Chosen 1

# Ignore objects which have distance more than MAX_DISTANCE: obj distance > MAX_DISTANCE
# Value of zero to disable filtering
MAX_DISTANCE = 13

# Saving parameters
WRITE_IMG = True
WRITE_TO_CSV = True

# Timers parameters
TIME_WINDOW = 200

o_class_mapping = {0: 'noise', 1: 'pedestrian', 2: 'cyclist', 3: 'vehicle'}
