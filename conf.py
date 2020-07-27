import camera_parameters as cp

# Logging parameters
LOG_LEVEL = 10

# Camera parameters
# sc_name = 'scene_1_TZK'
sc_name = 'lamp_pole_1'
scene = cp.scene[sc_name]
cam = scene['cam']

ANGLE = scene['angle']
HEIGHT = scene['height']
RES = scene['img_res_cap']
FPS = 10

CAM_PARAM_DIR = './camera_parameters/rpi'

# Classifier path
# CLF_PATH = 'clf_model/detailed_separate_clf_dict.pcl'
CLF_PATH = 'clf_model/lamp_pole_1.pcl'

DEVICE = '/home/ivan/NextCloudEs/experiments_data/sources/lighting_pole_1/vid_3_cars_selected/car_night_merged_rawvideo_gray.mkv'
# OUT_DIR = '/home/ivan/NextCloudEs/experiments_data/sources/lighting_pole_1/results/car_night_merged_temp_{}x{}'.format(RES[0], RES[1])
OUT_DIR = '/tmp/car_night_merged_temp_1{}x{}'.format(RES[0], RES[1])



# Pre-processing parameters
CLAHE_LIMIT = 3         # CLAHE contrast adjustment for grayscale images only (COLOR = 0)

# Background subtraction parameters
BGS_METHOD = 'MOG2'  # KNN is also available
BG_THR = 16  # 16  # For MOG2 only
DILATE_ITERATIONS = 1
HISTORY = 50
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
SAVER = False

# Timers parameters
TIME_WINDOW = 200

o_class_mapping = {0: 'noise', 1: 'pedestrian', 2: 'cyclist', 3: 'vehicle'}

SND_RECV_PORTS = (65433, 65434)  # Send and receive ports for interconnection with SmartLighting app
