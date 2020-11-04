# Description
The current repository is an implementation of detection method for low-performance hardware.
The method is used for detection of pedestrians, cyclists and vehicles in city environment.
The method is based on analysis of geometrical object features in a foreground mask. The foreground mask is obtained using background subtraction algorithm.
Classification is performed using logistic regression classifier.

## Prerequisites
The method can be used **only** when following conditions are satisfied:
1) Known intrinsic and extrinsic camera parameters.
2) The camera is mounted on a static object.
3) Trained classifier for a particular usage scenario. The training uses 3D object models and camera parameters on input.

## Usage
If the trained classifier is **already existing** and the camera has been **calibrated**, the algorithm can be run via:
```
python3 run_detection.py -p path_to_config.yml
```  
By default `configs/config.yml` will be used.

## Project structure
Below are shown subdirectories only. 

    .
    ├── clf                             # (*m) Classifiers
    │   ├── clf_name_1.pcl              # Name must match the one specified in config. Pickled sklearn object.
    │   └── ...
    ├── configs                         # Configuration files examples 
    │   ├── config.yml                  # (*m) Default configuration file
    │   └── ...
    ├── demo                            # Video and information for testing
    ├── prepare_img                     # Usefull scripts
    └── requirements                    # Python packages lists

***m - the file/directory name must match the given example**