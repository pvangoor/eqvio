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
./build/eqvio_opt ~/datasets/V1_01_easy/ configs/EQVIO_config_EuRoC_stationary.yaml --display --mode asl
```

This will create an output directory with a timestamp in the current directory, containing csv files with all the different states of eqvio.
The `--display` flag is required to create live animations of the VIO state estimate.

### Running and Analysing Datasets

There are a few python scripts provided to help with running and evaluating datasets.
These are meant for evaluation of EqVIO over common robotics datasets, particularly over the EuRoC and UZHFPV datasets.
They can also be used for some other datasets compatible with EqVIO.

Suppose you want to run EqVIO over the EuRoC sequence V1_01_easy and analyse the results you could use:

```
python3 scripts/run_and_analyse_dataset.py --mode=asl configs/EQVIO_config_EuRoC_stationary.yaml ~/datasets/V1_01_easy/
```

This will first run EqVIO over the dataset and then analyse the results.
Note that the `--display` option is not used and the usual EqVIO commandline output is suppressed, so there is no output until the running and analysis are complete.
The results are then saved to a new directory created in the dataset directory. In this case, `~/datasets/V1_01_easy/results/`.
The results directory includes a range of useful and interesting figures and information:

- **biases.pdf**: The estimated gyroscope and accelerometer biases over time.
- **camera_offset.pdf**: The estimated rotation and position of the camera frame relative to the IMU frame over time.
- **features.pdf**: The number of features at any given time and the mean lifetime of all the features in the state over time.
- **gravity_and_velocity.pdf**: The acceleration due to gravity and the linear velocity of the IMU, both expressed in the IMU frame, over time. 
- **results.yaml**: The numeric results of the algorithm, including position and attitude error statistics, scale error, mean processing time, etc.
- **timing_boxplots.pdf**: A boxplot of the time taken for each step in the algorithm.
- **timing_flamegraph.pdf**: A flamegraph showing how much time was spent on each step in the algorithm over time.
- **timing_histograms.pdf**: Histograms showing the distribution of time spent on each step of the algorithm.
- **trajectory_error.pdf**: The attitude and position errors over time, including x,y,z components as well as the norm.
- **trajectory.pdf**: The true and estimated trajectories over time in terms of x,y,z position and Euler angles (roll, pitch, yaw).
- **trajectory_xy.pdf**: A top-down (with respect to the z-axis of the true trajectory) view of the true and estimated trajectories.

If no groundtruth is available, then the plots are all still generated, except for the trajectory_error.pdf.
The true trajectory is obviously not shown in this case, since it is not known.

#### Running Multiple Datasets

The python scripts can also handle multiple sequences from the same dataset.
Simply put the names of all the sequences you want to test in the command:

```
python3 scripts/run_and_analyse_dataset.py --mode=asl configs/EQVIO_config_EuRoC_stationary.yaml ~/datasets/V1_01_easy/ ~/datasets/V1_02_medium/
```

This will generate results for each dataset.
These results can then be summarised into one file:

```
python3 scripts/summarise_results.py --mode=asl ~/datasets/V1_01_easy/ ~/datasets/V1_02_medium/
```

This generates a new file which collects all the most important results in one place for each dataset, such as position and attitude RMSE, mean processing time, and scale error.

The other way to run multiple datasets is to create a yaml file with information on the datasets you wish to analyse.
An example is provided in `scripts/euroc_sequences.yaml`.
As shown, this also allows you to add information such as the desired dataset mode and start time.
To run over all the EuRoC datasets and analyse the results:

```
python3 scripts/run_and_analyse_dataset.py configs/EQVIO_config_EuRoC_stationary.yaml scripts/euroc_sequences.yaml
python3 scripts/summarise_results.py scripts/euroc_sequences.yaml
```
