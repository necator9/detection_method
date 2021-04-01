# Description
The current repository is an implementation of detection method for low-performance Linux single-board computers.
The method is used for detection of pedestrians, cyclists and vehicles in city environment.
The method is based on analysis of geometrical object features in a foreground mask. The foreground mask is obtained using background subtraction algorithm.
Classification is performed using logistic regression classifier.
Implementation of the method is based on the publication [“Fast Object Detection Using Dimensional Based Features for Public Street Environments”](https://www.mdpi.com/2624-6511/3/1/6).

## Prerequisites
The method can be used **only** when following conditions are satisfied:
1) Known intrinsic and extrinsic (angle about X axis and height of installation) camera parameters.
2) The camera is mounted on a static object.
3) The [trained classifier](https://github.com/necator9/model_training) for a particular usage scenario. The training uses 3D object models and camera parameters on input.

## Usage
```
usage: run_detection.py [-h] [-c CLF] config

Run the lightweight detection algorithm

positional arguments:
  config             path to the configuration file

optional arguments:
  -h, --help         show this help message and exit
  -c CLF, --clf CLF  override path to the pickled classifier file given in
                     config
```

If the trained classifier is **already existing** and the camera has been **calibrated**, the algorithm can be run via:
```
python3 run_detection.py path_to_config.yml
```  

## Config structure

| key | type | description |
|---|---|---|
| log_level | int | logging level: 10 - debug, 50 - critical |
| device | str | path to video device or file  |
| resolution | list of ints | resolution (width, height) used for processing (1 - set as capturing resolution if it is supported by driver; 2 - resize captured frame to this resolution before processing) |
| fps | int | set capturing frame-per-second parameter if it is supported by driver |
| angle | int | camera incline towards to ground surface: 0 deg. - the camera is parallel to the ground surface; -90 deg. - camera points perpendicularly down |
| height | int | ground surface coordinates in meters relatively to a camera origin (e.g. -5 is 5m of camera height) |
| focal_length | int | camera focal length in mm |
| clf | str | path to the object containing the trained classifier |
| out_dir | str | output data such as logs, detection images, detection csv data will be stored in this directory |
| save_img | bool | enable or disable saving detection images with bounding rectangles and probabilities |
| save_csv | bool | enable or disable saving per-object detection information |
| stream | dict | define streaming parameters by keys: *enabled* - boolean to enable or disable streaming to rtsp streaming server, *server* - address:port of a rtsp streaming server   |
| clahe_limit | int | value of contrast limiting for adaptive histogram equalization (see [CLAHE](https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html))
| bgs_method | dict | parameters of background subtraction method defined by keys: *name* - name of the method available in OpenCV ([MOG2](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html), [KNN](https://docs.opencv.org/3.4/db/d88/classcv_1_1BackgroundSubtractorKNN.html) or [CNT](https://docs.opencv.org/3.4/db/d88/classcv_1_1BackgroundSubtractorKNN.html))  , *parameters* - list of parameters passed to method constructor |
| dilate_it | int | number of times [dilation](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html) is applied (dilates an image by using a specific structuring element) |
| time_window | int | number of samples defining periodicity of detection information printing (FPS, number of detections) |
| o_class_mapping | dict | mapping the integer object class to its string name of the following format: {class(int): class(string)} |
| sl_conn | dict | specifies parameters of connection to SL daemon by keys:  *detect_port* - the port on which the detection application receives for messages from SL daemon, *sl_port* - the port of SL daemon to which the detection application sends messages, *notif_interval* - interval in seconds defining frequency of notifications sending to SL daemon |
| lamp_on_criteria | list of ints | sends signal to switch on lamp when the criteria is satisfied. List [q, N]: On how many N frames out of the last q frames target objects have been detected |
| lamp_switching_time | int | time required for a physical lamp to change its state |
| cont_area_thr | float | filter out small objects having contour area lower than the threshold (cont_area_thr): object_contour_area / (RES[0] * RES[1]) > cont_area_thr, set 0 to disable |
| extent_thr | float | filter out objects having extent lower than the threshold (extent_thr), set 0 to disable |
| max_distance | float | filter out objects detected on distances more than the threshold (max_distance), set 0 to disable | 
| margin | int|  filter out objects intersecting frame margin (left, right, up, down), set 0 to disable |
| base_res | list of ints | resolution used for calibration |
| camera_matrix |  2D list of floats | camera intrinsics obtained via camera calibration |
| dist_coefs | 2D list of floats | radial and tangential camera lens distortion coefficients |
| optimized_res | list of ints | resolution used for points reprojection after lens distortions removal |
| optimized_matrix | 2D list of floats | camera intrinsics for points reprojection after lens distortions removal |

## Project structure

    .
    ├── demo                            # Data for testing
    │   ├── clf                         # Classifiers
    │   │   ├── clf_name_1.pcl          # Pickled dictionary containing sklearn object
    │   │   └── ...
    │   ├── configs                         # Configuration files examples 
    │   │   ├── config.yml                  
    │   │   └── ...
    ├── prepare_img                     # Usefull scripts
    └── requirements                    # Python packages lists

## Related
1. [Camera calibration](doc/calibration.md)
2. [Camera matrix optimization](https://github.com/necator9/get_optimal_cam_mtx)
3. [Classifier training](https://github.com/necator9/model_training)
4. [Bundle into a single executable](doc/pyinstaller.md)
5. [Streaming server](doc/streaming_server.md)

## Cite

I. Matveev, K. Karpov, I. Chmielewski, E. Siemens, and A. Yurchenko, “Fast Object Detection Using Dimensional Based Features for Public Street Environments,” Smart Cities, vol. 3, no. 1, Art. no. 1, Mar. 2020, doi: 10.3390/smartcities3010006.