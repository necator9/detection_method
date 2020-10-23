# Description
The current repository is an implementation of detection method for low-performance, linux-compatible hardware.
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

    .
    ├── build                   # Compiled files (alternatively `dist`)
    ├── docs                    # Documentation files (alternatively `doc`)
    ├── src                     # Source files (alternatively `lib` or `app`)
    ├── test                    # Automated tests (alternatively `spec` or `tests`)
    ├── tools                   # Tools and utilities
    ├── LICENSE
    └── README.md