# EqVIO: An Equivariant Filter for Visual Inertial Odometry {#mainpage}

This repository contains the implementation of EqVIO: An Equivariant Filter (EqF) for Visual Inertial Odometry (VIO).

## Dependencies

- Eigen 3: `sudo apt install libeigen3-dev`
- Yaml-cpp: `sudo apt install libyaml-cpp-dev`
- GIFT: https://github.com/pvangoor/GIFT

### Optional Dependencis

- FreeGLUT (for visualisations): `sudo apt install freeglut3-dev`
- ROS (for reading ROS-Bags): http://wiki.ros.org/ROS/Installation
- Doxygen (for documentation): `sudo apt install doxygen`

## Build Overview

EqVIO is designed to be built as a cmake project rather than a ROS package.
Assuming all prerequisites are installed and you are in the root folder of the repository, then you can follow these steps to build:

```
mkdir build
cd build
cmake ..
cmake --build . -j8
```

Note: on older machines, it may be better to use `-j4` or even `-j2` instead of `-j8`.
There are a number of flags that can be passed to cmake to influence the build process.
The key flags related to eqvio are all prefixed `EQVIO_*`.
For example, if you wish to build the documentation yourself, you could use `cmake .. -DEQVIO_BUILD_DOCS=1` instead of `cmake ..`.

### Common Issues

Some of the geometry functions use c++20 concepts.
If you cannot install a c++20 compatible compiler, then your other option is to disable concepts with `-DEQVIO_SUPPORT_CONCEPTS=0`.
Note that this project still uses c++17 features, and there are no plans to create workarounds.

## Usage Overview

The build process will create an executable called `eqvio_opt` in the build directory.
To see the help, simply run `./build/eqvio_opt --help`.
The required arguments are the dataset file/directory name, and the eqvio configuration file.
There are also many optional arguments to adjust how the program runs.
Different types of datasets are distinguished by providing a value to `--mode` (see the datasetReader page).
An example command is

```
./build/eqvio_opt /home/pieter/datasets/V1_01_easy/ configs/EQVIO_config_EuRoC_stationary.yaml --display --mode asl
```

This will create an output directory with a timestamp in the current directory, containing csv files with all the different states of eqvio.
The `--display` flag is required to create live animations of the VIO state estimate.