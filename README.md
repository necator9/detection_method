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
usage: run_detection [-h] [-p PATH] [-c CLF]

Run the lightweight detection algorithm

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  path to the configuration file (default: ./configs/config.yml)
  -c CLF, --clf CLF     path to the pickled classifier file (default: ./demo/clf/lamp_pole_1.pcl)
```

If the trained classifier is **already existing** and the camera has been **calibrated**, the algorithm can be run via:
```
python3 run_detection.py -p path_to_config.yml
```  

## Project structure

    .
    ├── configs                         # Configuration files examples 
    │   ├── config.yml                  # Default configuration file
    │   └── ...
    ├── demo                            # Data for testing
    │   ├── clf                         # Classifiers
    │       ├── clf_name_1.pcl          # Pickled sklearn object.
    │       └── ...
    ├── prepare_img                     # Usefull scripts
    └── requirements                    # Python packages lists

## Related
1. [Camera calibration](doc/calibration.md)
2. [Bundle into a single executable](doc/pyinstaller.md)
