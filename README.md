# Description
The current repository is an implementation of detection method for low-performance hardware.
The method is used for detection of pedestrians, cyclists and vehicles in city environment.
The method is based on analysis of geometrical object features in a foreground mask. The foreground mask is obtained using background subtraction algorithm.
Classification is performed using logistic regression classifier.

## Prerequisites
The method can be used **only** when following conditions are satisfied:
1) Known intrinsic and extrinsic (angle about X axis and height of installation) camera parameters.
2) The camera is mounted on a static object.
3) Trained classifier for a particular usage scenario. The training uses 3D object models and camera parameters on input.

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

## Project structure

    .
    ├── demo                            # Data for testing
    │   ├── clf                         # Classifiers
    │   │   ├── clf_name_1.pcl          # Pickled sklearn object.
    │   │   └── ...
    │   ├── configs                         # Configuration files examples 
    │   │   ├── config.yml                  
    │   │   └── ...
    ├── prepare_img                     # Usefull scripts
    └── requirements                    # Python packages lists

## Related
1. [Camera calibration](doc/calibration.md)
2. [Classifier training](https://github.com/necator9/model_training)
3. [Bundle into a single executable](doc/pyinstaller.md)
4. [Streaming server](doc/streaming_server.md)
